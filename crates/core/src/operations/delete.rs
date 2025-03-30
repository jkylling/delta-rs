//! Delete records from a Delta Table that satisfy a predicate
//!
//! When a predicate is not provided then all records are deleted from the Delta
//! Table. Otherwise a scan of the Delta table is performed to mark any files
//! that contain records that satisfy the predicate. Once files are determined
//! they are rewritten without the records.
//!
//!
//! Predicates MUST be deterministic otherwise undefined behaviour may occur during the
//! scanning and rewriting phase.
//!
//! # Example
//! ```rust ignore
//! let table = open_table("../path/to/table")?;
//! let (table, metrics) = DeleteBuilder::new(table.object_store(), table.state)
//!     .with_predicate(col("col1").eq(lit(1)))
//!     .await?;
//! ````

use async_trait::async_trait;
use datafusion::dataframe::DataFrame;
use datafusion::datasource::provider_as_source;
use datafusion::error::Result as DataFusionResult;
use datafusion::execution::context::{SessionContext, SessionState};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::Expr;
use datafusion_common::{HashSet, ScalarValue};
use datafusion_expr::{lit, Extension, LogicalPlan, LogicalPlanBuilder, UserDefinedLogicalNode};
use datafusion_physical_plan::metrics::MetricBuilder;
use datafusion_physical_plan::ExecutionPlan;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::io;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt64Type};
use arrow_array::RecordBatch;
use bytes::{BufMut, Bytes, BytesMut};
use datafusion::execution::TaskContext;
use futures::future::BoxFuture;
use futures::StreamExt;
use itertools::Itertools;
use object_store::path::Path;
use object_store::{ObjectStore, PutPayload};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use parquet::file::properties::WriterProperties;
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::Serialize;

use super::cdc::should_write_cdc;
use super::datafusion_utils::Expression;
use super::transaction::{CommitBuilder, CommitProperties, PROTOCOL};
use super::Operation;
use crate::delta_datafusion::expr::fmt_expr_to_sql;
use crate::delta_datafusion::logical::MetricObserver;
use crate::delta_datafusion::physical::{find_metric_node, get_metric, MetricObserverExec};
use crate::delta_datafusion::planner::DeltaPlanner;
use crate::delta_datafusion::{
    find_files, get_path_column, register_store, DataFusionMixins, DeltaScanConfigBuilder,
    DeltaSessionContext, DeltaTableProvider, PATH_COLUMN,
};
use crate::errors::DeltaResult;
use crate::kernel::arrow::extract::ProvidesColumnByName;
use crate::kernel::{Action, Add, DeletionVectorDescriptor, Remove, StorageType};
use crate::logstore::LogStoreRef;
use crate::operations::write::execution::{write_execution_plan, write_execution_plan_cdc};
use crate::operations::write::WriterStatsConfig;
use crate::operations::CustomExecuteHandler;
use crate::protocol::DeltaOperation;
use crate::storage::ObjectStoreRef;
use crate::table::state::DeltaTableState;
use crate::{DeltaTable, DeltaTableError};

const SOURCE_COUNT_ID: &str = "delete_source_count";
const SOURCE_COUNT_METRIC: &str = "num_source_rows";

/// Delete Records from the Delta Table.
/// See this module's documentation for more information
pub struct DeleteBuilder {
    /// Which records to delete
    predicate: Option<Expression>,
    /// A snapshot of the table's state
    snapshot: DeltaTableState,
    /// Delta object store for handling data files
    log_store: LogStoreRef,
    /// Datafusion session state relevant for executing the input plan
    state: Option<SessionState>,
    /// Properties passed to underlying parquet writer for when files are rewritten
    writer_properties: Option<WriterProperties>,
    /// Commit properties and configuration
    commit_properties: CommitProperties,
    custom_execute_handler: Option<Arc<dyn CustomExecuteHandler>>,
    /// Whether to write deletion vectors
    deletion_vectors: bool,
}

#[derive(Default, Debug, Serialize)]
/// Metrics for the Delete Operation
pub struct DeleteMetrics {
    /// Number of files added
    pub num_added_files: usize,
    /// Number of files removed
    pub num_removed_files: usize,
    /// Number of rows removed
    pub num_deleted_rows: usize,
    /// Number of rows copied in the process of deleting files
    pub num_copied_rows: usize,
    /// Time taken to execute the entire operation
    pub execution_time_ms: u64,
    /// Time taken to scan the file for matches
    pub scan_time_ms: u64,
    /// Time taken to rewrite the matched files
    pub rewrite_time_ms: u64,
}

impl super::Operation<()> for DeleteBuilder {
    fn log_store(&self) -> &LogStoreRef {
        &self.log_store
    }
    fn get_custom_execute_handler(&self) -> Option<Arc<dyn CustomExecuteHandler>> {
        self.custom_execute_handler.clone()
    }
}

impl DeleteBuilder {
    /// Create a new [`DeleteBuilder`]
    pub fn new(log_store: LogStoreRef, snapshot: DeltaTableState) -> Self {
        Self {
            predicate: None,
            snapshot,
            log_store,
            state: None,
            commit_properties: CommitProperties::default(),
            writer_properties: None,
            custom_execute_handler: None,
            deletion_vectors: false,
        }
    }

    /// A predicate that determines if a record is deleted
    pub fn with_predicate<E: Into<Expression>>(mut self, predicate: E) -> Self {
        self.predicate = Some(predicate.into());
        self
    }

    /// The Datafusion session state to use
    pub fn with_session_state(mut self, state: SessionState) -> Self {
        self.state = Some(state);
        self
    }

    /// Additional information to write to the commit
    pub fn with_commit_properties(mut self, commit_properties: CommitProperties) -> Self {
        self.commit_properties = commit_properties;
        self
    }

    /// Writer properties passed to parquet writer for when files are rewritten
    pub fn with_writer_properties(mut self, writer_properties: WriterProperties) -> Self {
        self.writer_properties = Some(writer_properties);
        self
    }

    /// Set a custom execute handler, for pre and post execution
    pub fn with_custom_execute_handler(mut self, handler: Arc<dyn CustomExecuteHandler>) -> Self {
        self.custom_execute_handler = Some(handler);
        self
    }

    /// Whether to write deletion vectors
    pub fn with_deletion_vectors(mut self, deletion_vectors: bool) -> Self {
        self.deletion_vectors = deletion_vectors;
        self
    }
}

#[derive(Clone, Debug)]
struct DeleteMetricExtensionPlanner {}

#[async_trait]
impl ExtensionPlanner for DeleteMetricExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> DataFusionResult<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(metric_observer) = node.as_any().downcast_ref::<MetricObserver>() {
            if metric_observer.id.eq(SOURCE_COUNT_ID) {
                return Ok(Some(MetricObserverExec::try_new(
                    SOURCE_COUNT_ID.into(),
                    physical_inputs,
                    |batch, metrics| {
                        MetricBuilder::new(metrics)
                            .global_counter(SOURCE_COUNT_METRIC)
                            .add(batch.num_rows());
                    },
                )?));
            }
        }
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_non_empty_expr(
    snapshot: &DeltaTableState,
    log_store: LogStoreRef,
    state: &SessionState,
    expression: &Expr,
    rewrite: &[Add],
    metrics: &mut DeleteMetrics,
    writer_properties: Option<WriterProperties>,
    partition_scan: bool,
    operation_id: Uuid,
) -> DeltaResult<Vec<Action>> {
    // For each identified file perform a parquet scan + filter + limit (1) + count.
    // If returned count is not zero then append the file to be rewritten and removed from the log. Otherwise do nothing to the file.
    let mut actions: Vec<Action> = Vec::new();
    let table_partition_cols = snapshot.metadata().partition_columns.clone();

    let delete_planner = DeltaPlanner::<DeleteMetricExtensionPlanner> {
        extension_planner: DeleteMetricExtensionPlanner {},
    };

    let state = SessionStateBuilder::new_from_existing(state.clone())
        .with_query_planner(Arc::new(delete_planner))
        .build();

    let scan_config = DeltaScanConfigBuilder::default()
        .with_file_column(false)
        .with_schema(snapshot.input_schema()?)
        .build(snapshot)?;

    let target_provider = Arc::new(
        DeltaTableProvider::try_new(snapshot.clone(), log_store.clone(), scan_config.clone())?
            .with_files(rewrite.to_vec()),
    );
    let target_provider = provider_as_source(target_provider);
    let source = LogicalPlanBuilder::scan("target", target_provider.clone(), None)?.build()?;

    let source = LogicalPlan::Extension(Extension {
        node: Arc::new(MetricObserver {
            id: "delete_source_count".into(),
            input: source,
            enable_pushdown: false,
        }),
    });

    let df = DataFrame::new(state.clone(), source);

    let writer_stats_config = WriterStatsConfig::new(
        snapshot.table_config().num_indexed_cols(),
        snapshot
            .table_config()
            .stats_columns()
            .map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<String>>()),
    );

    if !partition_scan {
        // Apply the negation of the filter and rewrite files
        let negated_expression = Expr::Not(Box::new(Expr::IsTrue(Box::new(expression.clone()))));

        let filter = df
            .clone()
            .filter(negated_expression)?
            .create_physical_plan()
            .await?;

        let add_actions: Vec<Action> = write_execution_plan(
            Some(snapshot),
            state.clone(),
            filter.clone(),
            table_partition_cols.clone(),
            log_store.object_store(Some(operation_id)),
            Some(snapshot.table_config().target_file_size() as usize),
            None,
            writer_properties.clone(),
            writer_stats_config.clone(),
        )
        .await?;

        actions.extend(add_actions);

        let source_count = find_metric_node(SOURCE_COUNT_ID, &filter).ok_or_else(|| {
            DeltaTableError::Generic("Unable to locate expected metric node".into())
        })?;
        let source_count_metrics = source_count.metrics().unwrap();
        let read_records = get_metric(&source_count_metrics, SOURCE_COUNT_METRIC);
        let filter_records = filter.metrics().and_then(|m| m.output_rows()).unwrap_or(0);

        metrics.num_copied_rows = filter_records;
        metrics.num_deleted_rows = read_records - filter_records;
    }

    actions.extend(
        write_cdc(
            snapshot,
            log_store.clone(),
            &expression,
            &state,
            writer_properties,
            operation_id,
            df,
        )
        .await?,
    );

    Ok(actions)
}

#[allow(clippy::too_many_arguments)]
async fn execute_deletion_vectors(
    snapshot: &DeltaTableState,
    log_store: LogStoreRef,
    state: &SessionState,
    expression: &Expr,
    rewrite: &[Add],
    metrics: &mut DeleteMetrics,
    writer_properties: Option<WriterProperties>,
    operation_id: Uuid,
) -> DeltaResult<Vec<Action>> {
    let mut actions: Vec<Action> = Vec::new();
    let table_partition_cols = snapshot.metadata().partition_columns.clone();

    let delete_planner = DeltaPlanner::<DeleteMetricExtensionPlanner> {
        extension_planner: DeleteMetricExtensionPlanner {},
    };

    let state = SessionStateBuilder::new_from_existing(state.clone())
        .with_query_planner(Arc::new(delete_planner))
        .build();

    let scan_config = DeltaScanConfigBuilder::default()
        .with_file_column(true)
        .with_row_number_column(Some(ROW_NUMBER_COLUMN.to_string()))
        .with_schema(snapshot.input_schema()?)
        .build(snapshot)?;

    let target_provider = Arc::new(
        DeltaTableProvider::try_new(snapshot.clone(), log_store.clone(), scan_config.clone())?
            .with_files(rewrite.to_vec()),
    );
    use datafusion::datasource::TableProvider;
    println!(
        "delete.arget_provider_schema: {:#?}",
        target_provider.schema()
    );
    let target_provider = provider_as_source(target_provider);
    let source = LogicalPlanBuilder::scan("target", target_provider.clone(), None)?.build()?;

    let source = LogicalPlan::Extension(Extension {
        node: Arc::new(MetricObserver {
            id: "delete_source_count".into(),
            input: source,
            enable_pushdown: false,
        }),
    });

    let df = DataFrame::new(state.clone(), source);

    let writer_stats_config = WriterStatsConfig::new(
        snapshot.table_config().num_indexed_cols(),
        snapshot
            .table_config()
            .stats_columns()
            .map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<String>>()),
    );

    // Apply the filter and rewrite files
    let filter_expression = Expr::IsTrue(Box::new(expression.clone()));

    println!("here1");
    let filter = df
        .clone()
        .filter(filter_expression)?
        .create_physical_plan()
        .await?;
    println!("here2");
    let add_actions: Vec<Action> = deletion_vectors_execution_plan(
        snapshot,
        state.clone(),
        filter.clone(),
        // table_partition_cols.clone(),
        log_store.object_store(Some(operation_id)),
        // None,
    )
    .await?;

    actions.extend(add_actions);

    let source_count = find_metric_node(SOURCE_COUNT_ID, &filter)
        .ok_or_else(|| DeltaTableError::Generic("Unable to locate expected metric node".into()))?;
    let source_count_metrics = source_count.metrics().unwrap();
    let read_records = get_metric(&source_count_metrics, SOURCE_COUNT_METRIC);
    let filter_records = filter.metrics().and_then(|m| m.output_rows()).unwrap_or(0);

    metrics.num_copied_rows = filter_records;
    metrics.num_deleted_rows = read_records - filter_records;

    actions.extend(
        write_cdc(
            snapshot,
            log_store.clone(),
            &expression,
            &state,
            writer_properties,
            operation_id,
            df,
        )
        .await?,
    );
    Ok(actions)
}

type DeletionVectorMap = HashMap<String, RoaringBitmap>;

/// In-memory writer for deletion vectors
struct DeletionVectorWriter {
    object_store: ObjectStoreRef,
    deletion_vectors: DeletionVectorMap,
}

const ROW_NUMBER_COLUMN: &str = "__delta_rs_row_number";

impl DeletionVectorWriter {
    fn new(object_store: ObjectStoreRef) -> Self {
        Self {
            object_store,
            deletion_vectors: Default::default(),
        }
    }

    fn write(&mut self, batch: &RecordBatch) -> DeltaResult<()> {
        let row_numbers = batch
            .column_by_name(ROW_NUMBER_COLUMN)
            .ok_or_else(|| {
                DeltaTableError::Generic(format!(
                    "Column '{}' not found in record batch",
                    ROW_NUMBER_COLUMN
                ))
            })?
            .as_primitive::<UInt64Type>();
        let file_dictionary = get_path_column(&batch, PATH_COLUMN)?;
        let mut file_names = file_dictionary.into_iter();
        let mut row_numbers = row_numbers.iter();
        let mut groups: HashMap<&str, Vec<u32>> = HashMap::new();
        while let (Some(row_number), Some(file_name)) = (row_numbers.next(), file_names.next()) {
            // TODO: Check u32
            groups
                .entry(file_name.unwrap())
                .or_insert_with(Vec::new)
                .push(row_number.unwrap() as u32);
        }
        for (file_name, row_numbers) in groups {
            // TODO: Avoid to_string
            self.deletion_vectors
                .entry(file_name.to_string())
                .or_insert_with(RoaringBitmap::new)
                .extend(row_numbers);
        }
        Ok(())
    }

    fn close(self) -> DeltaResult<DeletionVectorMap> {
        Ok(self.deletion_vectors)
    }
}

async fn deletion_vectors_execution_plan(
    snapshot: &DeltaTableState,
    state: SessionState,
    plan: Arc<dyn ExecutionPlan>,
    object_store: ObjectStoreRef,
) -> DeltaResult<Vec<Action>> {
    let mut tasks = vec![];
    for i in 0..plan.properties().output_partitioning().partition_count() {
        // Note: A single file can be split over multiple partitions. This is good, but means that we must run a final reduction step over the writers
        let inner_plan = plan.clone();
        let task_ctx = Arc::new(TaskContext::from(&state));
        let mut stream = inner_plan.execute(i, task_ctx)?;

        let mut writer = DeletionVectorWriter::new(object_store.clone());

        let handle: tokio::task::JoinHandle<DeltaResult<DeletionVectorMap>> =
            tokio::task::spawn(async move {
                while let Some(maybe_batch) = stream.next().await {
                    let batch = maybe_batch?;
                    writer.write(&batch)?;
                }
                writer.close()
            });
        tasks.push(handle);
    }

    let deletion_vectors = futures::future::join_all(tasks)
        .await
        .into_iter()
        .collect::<Result<Result<Vec<_>, _>, _>>()
        .map_err(|err| DeltaTableError::Generic(err.to_string()))??;
    let Some(deletion_vectors) = deletion_vectors.into_iter().reduce(|mut result, part| {
        for (k, v) in part {
            match result.entry(k) {
                Entry::Occupied(mut occupied) => occupied.get_mut().extend(&v),
                Entry::Vacant(mut vacant) => {
                    vacant.insert(v);
                }
            }
        }
        result
    }) else {
        return Ok(vec![]);
    };

    // TODO: Add read support
    // TODO: Test with partition columns
    // TODO: Merge with existing deletion vectors if they exist. That is, make an initial pass over touched files, read their deletion vectors (in parallel) and merge them with the new deletion vectors.
    // Write deletion vector files
    println!("deletion_vectors: {:#?}", deletion_vectors);
    let deletion_vector_descriptors =
        write_deletion_vectors_object_store(object_store, deletion_vectors).await?;
    let mut actions = vec![];
    for mut add in snapshot.snapshot.file_actions()? {
        if let Some(deletion_vector) = deletion_vector_descriptors.get(&add.path) {
            let remove = Action::Remove(Remove {
                path: add.path.clone(),
                data_change: true,
                ..Default::default()
            });
            add.deletion_vector = Some(deletion_vector.clone());
            actions.push(remove);
            actions.push(Action::Add(add));
        }
    }

    Ok(actions)
}

async fn write_deletion_vectors_object_store(
    object_store: ObjectStoreRef,
    deletion_vector_map: DeletionVectorMap,
) -> DeltaResult<HashMap<String, DeletionVectorDescriptor>> {
    let uuid = Uuid::new_v4();
    let path = Path::parse(format!("deletion_vector_{}.bin", uuid)).expect("Invalid path");
    let (result, bytes) = write_deletion_vectors_file(&uuid, deletion_vector_map)?;
    object_store
        .put(&path, PutPayload::from_bytes(bytes))
        .await?;
    println!("writing deletion vecotr to: {:?}", path);
    Ok(result)
}

const DELETION_VECTOR_MAGIC: [u8; 4] = 1681511377u32.to_be_bytes();
const DELETION_VECTOR_FILE_FORMAT_VERSION_1: u8 = 1;

mod size {
    pub const VERSION: usize = 1;
    pub const DATA_SIZE: usize = 4;
    pub const MAGIC: usize = 4;
    pub const CHECKSUM: usize = 4;
}

fn write_deletion_vectors_file<I>(
    uuid: &Uuid,
    deletion_vectors: I,
) -> DeltaResult<(HashMap<String, DeletionVectorDescriptor>, Bytes)>
where
    for<'a> &'a I: IntoIterator<Item = (&'a String, &'a RoaringBitmap)>,
    I: IntoIterator<Item = (String, RoaringBitmap)>,
{
    // We use UUID relative paths for deletion vectors (like the Spark implementation).
    // Protocol definition at: https://github.com/delta-io/delta/blob/master/PROTOCOL.md#deletion-vector-format
    // Spark reference implementation: https://github.com/delta-io/delta/blob/e7581526e3b7235621c2f4973c799e2f144350cf/spark/src/main/scala/org/apache/spark/sql/delta/storage/dv/DeletionVectorStore.scala#L213-L245
    let mut result = HashMap::new();
    let buffer_size = size::VERSION
        + (&deletion_vectors)
            .into_iter()
            .map(|(_, v)| size::DATA_SIZE + size::MAGIC + v.serialized_size() + size::CHECKSUM)
            .sum::<usize>();
    let mut buffer = BytesMut::with_capacity(buffer_size);
    buffer.put_u8(DELETION_VECTOR_FILE_FORMAT_VERSION_1);
    for (file_name, deletion_vector) in deletion_vectors.into_iter() {
        let PartialDeletionVectorDescriptor {
            offset,
            size_in_bytes,
            cardinality,
        } = write_deletion_vector(&mut buffer, &deletion_vector)?;
        let path_or_inline_dv = z85::encode(uuid.as_bytes());
        result.insert(
            file_name,
            DeletionVectorDescriptor {
                storage_type: StorageType::UuidRelativePath,
                path_or_inline_dv,
                offset: Some(offset),
                size_in_bytes,
                cardinality,
            },
        );
    }
    Ok((result, buffer.freeze()))
}

struct PartialDeletionVectorDescriptor {
    offset: i32,
    size_in_bytes: i32,
    cardinality: i64,
}

fn write_deletion_vector(
    buffer: &mut BytesMut,
    deletion_vector: &RoaringBitmap,
) -> io::Result<PartialDeletionVectorDescriptor> {
    let offset = buffer.len();
    let cardinality = deletion_vector.len() as i64;
    let size_in_bytes = size::MAGIC + deletion_vector.serialized_size();
    buffer.put_u32(size_in_bytes as u32);
    buffer.put_slice(&DELETION_VECTOR_MAGIC);
    deletion_vector.serialize_into(buffer.writer())?;
    let checksum = crc32fast::hash(
        &buffer[offset + size::DATA_SIZE..offset + size::DATA_SIZE + size_in_bytes],
    );
    buffer.put_u32(checksum);
    Ok(PartialDeletionVectorDescriptor {
        offset: offset as i32,
        cardinality,
        size_in_bytes: size_in_bytes as i32,
    })
}

async fn write_cdc(
    snapshot: &DeltaTableState,
    log_store: LogStoreRef,
    expression: &Expr,
    state: &SessionState,
    writer_properties: Option<WriterProperties>,
    operation_id: Uuid,
    df: DataFrame,
) -> DeltaResult<Vec<Action>> {
    let table_partition_cols = snapshot.metadata().partition_columns.clone();
    let writer_stats_config = WriterStatsConfig::new(
        snapshot.table_config().num_indexed_cols(),
        snapshot
            .table_config()
            .stats_columns()
            .map(|v| v.iter().map(|v| v.to_string()).collect::<Vec<String>>()),
    );

    // CDC logic, simply filters data with predicate and adds the _change_type="delete" as literal column
    if let Ok(true) = should_write_cdc(snapshot) {
        // Create CDC scan
        let change_type_lit = lit(ScalarValue::Utf8(Some("delete".to_string())));
        let cdc_filter = df
            .filter(expression.clone())?
            .with_column("_change_type", change_type_lit)?
            .create_physical_plan()
            .await?;

        return Ok(write_execution_plan_cdc(
            Some(snapshot),
            state.clone(),
            cdc_filter,
            table_partition_cols.clone(),
            log_store.object_store(Some(operation_id)),
            Some(snapshot.table_config().target_file_size() as usize),
            None,
            writer_properties,
            writer_stats_config,
        )
        .await?);
    }
    return Ok(vec![]);
}

#[allow(clippy::too_many_arguments)]
async fn execute(
    predicate: Option<Expr>,
    log_store: LogStoreRef,
    snapshot: DeltaTableState,
    state: SessionState,
    writer_properties: Option<WriterProperties>,
    mut commit_properties: CommitProperties,
    operation_id: Uuid,
    handle: Option<&Arc<dyn CustomExecuteHandler>>,
    deletion_vectors: bool,
) -> DeltaResult<(DeltaTableState, DeleteMetrics)> {
    if !&snapshot.load_config().require_files {
        return Err(DeltaTableError::NotInitializedWithFiles("DELETE".into()));
    }

    let exec_start = Instant::now();
    let mut metrics = DeleteMetrics::default();

    let scan_start = Instant::now();
    let candidates = find_files(&snapshot, log_store.clone(), &state, predicate.clone()).await?;
    metrics.scan_time_ms = Instant::now().duration_since(scan_start).as_millis() as u64;

    let predicate = predicate.unwrap_or(Expr::Literal(ScalarValue::Boolean(Some(true))));

    let mut actions = {
        let write_start = Instant::now();
        let add = if !deletion_vectors {
            execute_non_empty_expr(
                &snapshot,
                log_store.clone(),
                &state,
                &predicate,
                &candidates.candidates,
                &mut metrics,
                writer_properties,
                candidates.partition_scan,
                operation_id,
            )
            .await?
        } else if deletion_vectors && !candidates.partition_scan {
            execute_deletion_vectors(
                &snapshot,
                log_store.clone(),
                &state,
                &predicate,
                &candidates.candidates,
                &mut metrics,
                writer_properties,
                operation_id,
            )
            .await?
        } else {
            // TODO
            panic!()
        };
        metrics.rewrite_time_ms = Instant::now().duration_since(write_start).as_millis() as u64;
        add
    };
    let remove = candidates.candidates;

    let deletion_timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;

    metrics.num_removed_files = remove.len();
    metrics.num_added_files = actions.len();

    // TODO: Is it always correct to remove all candidates?
    for action in remove {
        actions.push(Action::Remove(Remove {
            path: action.path,
            deletion_timestamp: Some(deletion_timestamp),
            data_change: true,
            extended_file_metadata: Some(true),
            partition_values: Some(action.partition_values),
            size: Some(action.size),
            deletion_vector: action.deletion_vector,
            tags: None,
            base_row_id: action.base_row_id,
            default_row_commit_version: action.default_row_commit_version,
        }))
    }

    metrics.execution_time_ms = Instant::now().duration_since(exec_start).as_millis() as u64;

    commit_properties
        .app_metadata
        .insert("readVersion".to_owned(), snapshot.version().into());
    commit_properties.app_metadata.insert(
        "operationMetrics".to_owned(),
        serde_json::to_value(&metrics)?,
    );

    // Do not make a commit when there are zero updates to the state
    let operation = DeltaOperation::Delete {
        predicate: Some(fmt_expr_to_sql(&predicate)?),
    };
    if actions.is_empty() {
        return Ok((snapshot.clone(), metrics));
    }

    let commit = CommitBuilder::from(commit_properties)
        .with_actions(actions)
        .with_operation_id(operation_id)
        .with_post_commit_hook_handler(handle.cloned())
        .build(Some(&snapshot), log_store.clone(), operation)
        .await?;

    if let Some(handler) = handle {
        handler.post_execute(&log_store, operation_id).await?;
    }
    Ok((commit.snapshot(), metrics))
}

impl std::future::IntoFuture for DeleteBuilder {
    type Output = DeltaResult<(DeltaTable, DeleteMetrics)>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        let this = self;

        Box::pin(async move {
            PROTOCOL.check_append_only(&this.snapshot.snapshot)?;
            PROTOCOL.can_write_to(&this.snapshot.snapshot)?;

            let operation_id = this.get_operation_id();
            this.pre_execute(operation_id).await?;

            let state = this.state.unwrap_or_else(|| {
                let session: SessionContext = DeltaSessionContext::default().into();

                // If a user provides their own their DF state then they must register the store themselves
                register_store(this.log_store.clone(), session.runtime_env());

                session.state()
            });

            let predicate = match this.predicate {
                Some(predicate) => match predicate {
                    Expression::DataFusion(expr) => Some(expr),
                    Expression::String(s) => {
                        Some(this.snapshot.parse_predicate_expression(s, &state)?)
                    }
                },
                None => None,
            };

            let (new_snapshot, metrics) = execute(
                predicate,
                this.log_store.clone(),
                this.snapshot,
                state,
                this.writer_properties,
                this.commit_properties,
                operation_id,
                this.custom_execute_handler.as_ref(),
                this.deletion_vectors,
            )
            .await?;

            Ok((
                DeltaTable::new_with_state(this.log_store, new_snapshot),
                metrics,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::delta_datafusion::DeltaScanConfigBuilder;
    use crate::delta_datafusion::DeltaTableProvider;
    use crate::kernel::{DataType as DeltaDataType, StorageType};
    use crate::operations::collect_sendable_stream;
    use crate::operations::delete::DeletionVectorDescriptor;
    use crate::operations::delete::{write_deletion_vector, write_deletion_vectors_file};
    use crate::operations::DeltaOps;
    use crate::protocol::*;
    use crate::writer::test_utils::datafusion::get_data;
    use crate::writer::test_utils::datafusion::write_batch;
    use crate::writer::test_utils::{
        get_arrow_schema, get_delta_schema, get_record_batch, setup_table_with_configuration,
    };
    use crate::DeltaTable;
    use crate::TableProperty;
    use arrow::array::Array;
    use arrow::array::Int32Array;
    use arrow::datatypes::{Field, Schema};
    use arrow::record_batch::RecordBatch;
    use arrow_array::ArrayRef;
    use arrow_array::StringArray;
    use arrow_array::StructArray;
    use arrow_buffer::NullBuffer;
    use arrow_schema::DataType;
    use arrow_schema::Fields;
    use bytes::BufMut;
    use bytes::BytesMut;
    use datafusion::assert_batches_sorted_eq;
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::*;
    use delta_kernel::schema::PrimitiveType;
    use futures::TryStreamExt;
    use roaring::RoaringBitmap;
    use serde_json::json;
    use std::collections::{BTreeMap, HashMap};
    use std::io::Write;
    use std::sync::Arc;
    use uuid::Uuid;
    use arrow::array::AsArray;
    use object_store::ObjectStore;
    use object_store::path::Path;
    use url::Url;

    async fn setup_table(partitions: Option<Vec<&str>>) -> DeltaTable {
        let table_schema = get_delta_schema();

        let table = DeltaOps::new_in_memory()
            .create()
            .with_columns(table_schema.fields().cloned())
            .with_partition_columns(partitions.unwrap_or_default())
            .await
            .unwrap();
        assert_eq!(table.version(), 0);
        table
    }

    #[tokio::test]
    async fn test_delete_when_delta_table_is_append_only() {
        let table = setup_table_with_configuration(TableProperty::AppendOnly, Some("true")).await;
        let batch = get_record_batch(None, false);
        // append some data
        let table = write_batch(table, batch).await;
        // delete
        let _err = DeltaOps(table)
            .delete()
            .await
            .expect_err("Remove action is included when Delta table is append-only. Should error");
    }

    #[tokio::test]
    async fn test_delete_default() {
        let schema = get_arrow_schema(&None);
        let table = setup_table(None).await;

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["A", "B", "A", "A"])),
                Arc::new(arrow::array::Int32Array::from(vec![1, 10, 10, 100])),
                Arc::new(arrow::array::StringArray::from(vec![
                    "2021-02-02",
                    "2021-02-02",
                    "2021-02-02",
                    "2021-02-02",
                ])),
            ],
        )
            .unwrap();
        // write some data
        let table = DeltaOps(table)
            .write(vec![batch.clone()])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        assert_eq!(table.version(), 1);
        assert_eq!(table.get_files_count(), 1);

        let (table, metrics) = DeltaOps(table).delete().await.unwrap();

        assert_eq!(table.version(), 2);
        assert_eq!(table.get_files_count(), 0);
        assert_eq!(metrics.num_added_files, 0);
        assert_eq!(metrics.num_removed_files, 1);
        assert_eq!(metrics.num_deleted_rows, 0);
        assert_eq!(metrics.num_copied_rows, 0);

        let commit_info = table.history(None).await.unwrap();
        let last_commit = &commit_info[0];
        let _extra_info = last_commit.info.clone();
        // assert_eq!(
        //     extra_info["operationMetrics"],
        //     serde_json::to_value(&metrics).unwrap()
        // );

        // Deletes with no changes to state must not commit
        let (table, metrics) = DeltaOps(table).delete().await.unwrap();
        assert_eq!(table.version(), 2);
        assert_eq!(metrics.num_added_files, 0);
        assert_eq!(metrics.num_removed_files, 0);
        assert_eq!(metrics.num_deleted_rows, 0);
        assert_eq!(metrics.num_copied_rows, 0);
    }

    #[tokio::test]
    async fn test_delete_on_nonpartition_column() {
        // Delete based on a nonpartition column
        // Only rewrite files that match the predicate
        // Test data designed to force a scan of the underlying data

        let schema = get_arrow_schema(&None);
        let table = setup_table(None).await;

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["A", "B", "A", "A"])),
                Arc::new(arrow::array::Int32Array::from(vec![1, 10, 10, 100])),
                Arc::new(arrow::array::StringArray::from(vec![
                    "2021-02-02",
                    "2021-02-02",
                    "2021-02-02",
                    "2021-02-02",
                ])),
            ],
        )
            .unwrap();

        // write some data
        let table = DeltaOps(table)
            .write(vec![batch])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        assert_eq!(table.version(), 1);
        assert_eq!(table.get_files_count(), 1);

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["A", "B", "A", "A"])),
                Arc::new(arrow::array::Int32Array::from(vec![0, 20, 10, 100])),
                Arc::new(arrow::array::StringArray::from(vec![
                    "2021-02-02",
                    "2021-02-02",
                    "2021-02-02",
                    "2021-02-02",
                ])),
            ],
        )
            .unwrap();

        // write some data
        let table = DeltaOps(table)
            .write(vec![batch])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        assert_eq!(table.version(), 2);
        assert_eq!(table.get_files_count(), 2);

        let (table, metrics) = DeltaOps(table)
            .delete()
            .with_predicate(col("value").eq(lit(1)))
            .await
            .unwrap();
        assert_eq!(table.version(), 3);
        assert_eq!(table.get_files_count(), 2);

        assert_eq!(metrics.num_added_files, 1);
        assert_eq!(metrics.num_removed_files, 1);
        assert!(metrics.scan_time_ms > 0);
        assert_eq!(metrics.num_deleted_rows, 1);
        assert_eq!(metrics.num_copied_rows, 3);

        let commit_info = table.history(None).await.unwrap();
        let last_commit = &commit_info[0];
        let parameters = last_commit.operation_parameters.clone().unwrap();
        assert_eq!(parameters["predicate"], json!("value = 1"));

        let expected = vec![
            "+----+-------+------------+",
            "| id | value | modified   |",
            "+----+-------+------------+",
            "| A  | 0     | 2021-02-02 |",
            "| A  | 10    | 2021-02-02 |",
            "| A  | 10    | 2021-02-02 |",
            "| A  | 100   | 2021-02-02 |",
            "| A  | 100   | 2021-02-02 |",
            "| B  | 10    | 2021-02-02 |",
            "| B  | 20    | 2021-02-02 |",
            "+----+-------+------------+",
        ];

        let actual = get_data(&table).await;
        assert_batches_sorted_eq!(&expected, &actual);
    }

    #[tokio::test]
    async fn test_delete_null() {
        // Demonstrate deletion of null

        async fn prepare_table() -> DeltaTable {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "value",
                arrow::datatypes::DataType::Int32,
                true,
            )]));

            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(vec![
                    Some(0),
                    None,
                    Some(2),
                    None,
                    Some(4),
                ]))],
            )
                .unwrap();

            DeltaOps::new_in_memory().write(vec![batch]).await.unwrap()
        }

        // Validate behaviour of greater than
        let table = prepare_table().await;
        let (table, _) = DeltaOps(table)
            .delete()
            .with_predicate(col("value").gt(lit(2)))
            .await
            .unwrap();

        let expected = vec![
            "+-------+",
            "| value |",
            "+-------+",
            "|       |",
            "|       |",
            "| 0     |",
            "| 2     |",
            "+-------+",
        ];
        let actual = get_data(&table).await;
        assert_batches_sorted_eq!(&expected, &actual);

        // Validate behaviour of less than
        let table = prepare_table().await;
        let (table, _) = DeltaOps(table)
            .delete()
            .with_predicate(col("value").lt(lit(2)))
            .await
            .unwrap();

        let expected = vec![
            "+-------+",
            "| value |",
            "+-------+",
            "|       |",
            "|       |",
            "| 2     |",
            "| 4     |",
            "+-------+",
        ];
        let actual = get_data(&table).await;
        assert_batches_sorted_eq!(&expected, &actual);

        // Validate behaviour of less plus not null
        let table = prepare_table().await;
        let (table, _) = DeltaOps(table)
            .delete()
            .with_predicate(col("value").lt(lit(2)).or(col("value").is_null()))
            .await
            .unwrap();

        let expected = vec![
            "+-------+",
            "| value |",
            "+-------+",
            "| 2     |",
            "| 4     |",
            "+-------+",
        ];
        let actual = get_data(&table).await;
        assert_batches_sorted_eq!(&expected, &actual);
    }

    #[tokio::test]
    async fn test_delete_on_partition_column() {
        // Perform a delete where the predicate only contains partition columns

        let schema = get_arrow_schema(&None);
        let table = setup_table(Some(["modified"].to_vec())).await;

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["A", "B", "A", "A"])),
                Arc::new(arrow::array::Int32Array::from(vec![0, 20, 10, 100])),
                Arc::new(arrow::array::StringArray::from(vec![
                    "2021-02-02",
                    "2021-02-03",
                    "2021-02-02",
                    "2021-02-03",
                ])),
            ],
        )
            .unwrap();

        // write some data
        let table = DeltaOps(table)
            .write(vec![batch])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        assert_eq!(table.version(), 1);
        assert_eq!(table.get_files_count(), 2);

        let (table, metrics) = DeltaOps(table)
            .delete()
            .with_predicate(col("modified").eq(lit("2021-02-03")))
            .await
            .unwrap();
        assert_eq!(table.version(), 2);
        assert_eq!(table.get_files_count(), 1);

        assert_eq!(metrics.num_added_files, 0);
        assert_eq!(metrics.num_removed_files, 1);
        assert_eq!(metrics.num_deleted_rows, 0);
        assert_eq!(metrics.num_copied_rows, 0);
        assert!(metrics.scan_time_ms > 0);

        let expected = vec![
            "+----+-------+------------+",
            "| id | value | modified   |",
            "+----+-------+------------+",
            "| A  | 0     | 2021-02-02 |",
            "| A  | 10    | 2021-02-02 |",
            "+----+-------+------------+",
        ];

        let actual = get_data(&table).await;
        assert_batches_sorted_eq!(&expected, &actual);
    }

    #[tokio::test]
    async fn test_delete_on_mixed_columns() {
        // Test predicates that contain non-partition and partition column
        let schema = get_arrow_schema(&None);
        let table = setup_table(Some(["modified"].to_vec())).await;

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["A", "B", "A", "A"])),
                Arc::new(arrow::array::Int32Array::from(vec![0, 20, 10, 100])),
                Arc::new(arrow::array::StringArray::from(vec![
                    "2021-02-02",
                    "2021-02-03",
                    "2021-02-02",
                    "2021-02-04",
                ])),
            ],
        )
            .unwrap();

        // write some data
        let table = DeltaOps(table)
            .write(vec![batch])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        assert_eq!(table.version(), 1);
        assert_eq!(table.get_files_count(), 3);

        let (table, metrics) = DeltaOps(table)
            .delete()
            .with_predicate(
                col("modified")
                    .eq(lit("2021-02-04"))
                    .and(col("value").eq(lit(100))),
            )
            .await
            .unwrap();
        assert_eq!(table.version(), 2);
        assert_eq!(table.get_files_count(), 2);

        assert_eq!(metrics.num_added_files, 0);
        assert_eq!(metrics.num_removed_files, 1);
        assert_eq!(metrics.num_deleted_rows, 1);
        assert_eq!(metrics.num_copied_rows, 0);
        assert!(metrics.scan_time_ms > 0);

        let expected = [
            "+----+-------+------------+",
            "| id | value | modified   |",
            "+----+-------+------------+",
            "| A  | 0     | 2021-02-02 |",
            "| A  | 10    | 2021-02-02 |",
            "| B  | 20    | 2021-02-03 |",
            "+----+-------+------------+",
        ];
        let actual = get_data(&table).await;
        assert_batches_sorted_eq!(&expected, &actual);
    }

    #[tokio::test]
    async fn test_delete_nested() {
        use arrow_schema::{DataType, Field, Schema as ArrowSchema};
        // Test Delete with a predicate that references struct fields
        // See #2019
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Utf8, true),
            Field::new(
                "props",
                DataType::Struct(Fields::from(vec![Field::new("a", DataType::Utf8, true)])),
                true,
            ),
        ]));

        let struct_array = StructArray::new(
            Fields::from(vec![Field::new("a", DataType::Utf8, true)]),
            vec![Arc::new(arrow::array::StringArray::from(vec![
                Some("2021-02-01"),
                Some("2021-02-02"),
                None,
                None,
            ])) as ArrayRef],
            Some(NullBuffer::from_iter(vec![true, true, true, false])),
        );

        let data = vec![
            Arc::new(arrow::array::StringArray::from(vec!["A", "B", "C", "D"])) as ArrayRef,
            Arc::new(struct_array) as ArrayRef,
        ];
        let batches = vec![RecordBatch::try_new(schema.clone(), data).unwrap()];

        let table = DeltaOps::new_in_memory().write(batches).await.unwrap();

        let (table, _metrics) = DeltaOps(table)
            .delete()
            .with_predicate("props['a'] = '2021-02-02'")
            .await
            .unwrap();

        let expected = [
            "+----+-----------------+",
            "| id | props           |",
            "+----+-----------------+",
            "| A  | {a: 2021-02-01} |",
            "| C  | {a: }           |",
            "| D  |                 |",
            "+----+-----------------+",
        ];
        let actual = get_data(&table).await;
        assert_batches_sorted_eq!(&expected, &actual);
    }

    #[tokio::test]
    async fn test_failure_nondeterministic_query() {
        // Deletion requires a deterministic predicate

        let table = setup_table(None).await;

        let res = DeltaOps(table)
            .delete()
            .with_predicate(col("value").eq(cast(
                random() * lit(20.0),
                arrow::datatypes::DataType::Int32,
            )))
            .await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_delete_cdc_enabled() {
        let table: DeltaTable = DeltaOps::new_in_memory()
            .create()
            .with_column(
                "value",
                DeltaDataType::Primitive(PrimitiveType::Integer),
                true,
                None,
            )
            .with_configuration_property(TableProperty::EnableChangeDataFeed, Some("true"))
            .await
            .unwrap();
        assert_eq!(table.version(), 0);

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            arrow::datatypes::DataType::Int32,
            true,
        )]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)]))],
        )
            .unwrap();
        let table = DeltaOps(table)
            .write(vec![batch])
            .await
            .expect("Failed to write first batch");
        assert_eq!(table.version(), 1);

        let (table, _metrics) = DeltaOps(table)
            .delete()
            .with_predicate(col("value").eq(lit(2)))
            .await
            .unwrap();
        assert_eq!(table.version(), 2);

        let ctx = SessionContext::new();
        let table = DeltaOps(table)
            .load_cdf()
            .with_starting_version(0)
            .build(&ctx.state(), None)
            .await
            .expect("Failed to load CDF");

        let mut batches = collect_batches(
            table.properties().output_partitioning().partition_count(),
            table,
            ctx,
        )
            .await
            .expect("Failed to collect batches");

        // The batches will contain a current _commit_timestamp which shouldn't be check_append_only
        let _: Vec<_> = batches.iter_mut().map(|b| b.remove_column(3)).collect();

        assert_batches_sorted_eq! {[
        "+-------+--------------+-----------------+",
        "| value | _change_type | _commit_version |",
        "+-------+--------------+-----------------+",
        "| 1     | insert       | 1               |",
        "| 2     | delete       | 2               |",
        "| 2     | insert       | 1               |",
        "| 3     | insert       | 1               |",
        "+-------+--------------+-----------------+",
        ], &batches }
    }

    #[tokio::test]
    async fn test_delete_cdc_enabled_partitioned() {
        let table: DeltaTable = DeltaOps::new_in_memory()
            .create()
            .with_column(
                "year",
                DeltaDataType::Primitive(PrimitiveType::String),
                true,
                None,
            )
            .with_column(
                "value",
                DeltaDataType::Primitive(PrimitiveType::Integer),
                true,
                None,
            )
            .with_partition_columns(vec!["year"])
            .with_configuration_property(TableProperty::EnableChangeDataFeed, Some("true"))
            .await
            .unwrap();
        assert_eq!(table.version(), 0);

        let schema = Arc::new(Schema::new(vec![
            Field::new("year", DataType::Utf8, true),
            Field::new("value", DataType::Int32, true),
        ]));

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec![
                    Some("2020"),
                    Some("2020"),
                    Some("2024"),
                ])),
                Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])),
            ],
        )
            .unwrap();

        let table = DeltaOps(table)
            .write(vec![batch])
            .await
            .expect("Failed to write first batch");
        assert_eq!(table.version(), 1);

        let (table, _metrics) = DeltaOps(table)
            .delete()
            .with_predicate(col("value").eq(lit(2)))
            .await
            .unwrap();
        assert_eq!(table.version(), 2);

        let ctx = SessionContext::new();
        let table = DeltaOps(table)
            .load_cdf()
            .with_starting_version(0)
            .build(&ctx.state(), None)
            .await
            .expect("Failed to load CDF");

        let mut batches = collect_batches(
            table.properties().output_partitioning().partition_count(),
            table,
            ctx,
        )
            .await
            .expect("Failed to collect batches");

        // The batches will contain a current _commit_timestamp which shouldn't be check_append_only
        let _: Vec<_> = batches.iter_mut().map(|b| b.remove_column(4)).collect();

        assert_batches_sorted_eq! {[
            "+-------+------+--------------+-----------------+",
            "| value | year | _change_type | _commit_version |",
            "+-------+------+--------------+-----------------+",
            "| 1     | 2020 | insert       | 1               |",
            "| 2     | 2020 | delete       | 2               |",
            "| 2     | 2020 | insert       | 1               |",
            "| 3     | 2024 | insert       | 1               |",
            "+-------+------+--------------+-----------------+",
        ], &batches }
    }

    async fn collect_batches(
        num_partitions: usize,
        stream: Arc<dyn ExecutionPlan>,
        ctx: SessionContext,
    ) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
        let mut batches = vec![];
        for p in 0..num_partitions {
            let data: Vec<RecordBatch> =
                collect_sendable_stream(stream.execute(p, ctx.task_ctx())?).await?;
            batches.extend_from_slice(&data);
        }
        Ok(batches)
    }

    #[test]
    fn test_write_deletion_vectors_file() {
        use roaring::RoaringBitmap;
        let uuid = Uuid::from_u128(1);
        let (result, bytes) = write_deletion_vectors_file(
            &uuid,
            BTreeMap::from_iter([
                (
                    "file1.parquet".to_string(),
                    RoaringBitmap::from_iter(vec![1, 2, 3]),
                ),
                (
                    "file2.parquet".to_string(),
                    RoaringBitmap::from_iter(vec![4, 5, 6]),
                ),
            ]),
        )
            .expect("Failed to write deletion vectors file");
        assert_eq!(result.len(), 2);
        assert_eq!(bytes.len(), 69);

        let file1 = result.get("file1.parquet").unwrap();
        assert_eq!(
            file1,
            &DeletionVectorDescriptor {
                storage_type: StorageType::UuidRelativePath,
                path_or_inline_dv: z85::encode(uuid.as_bytes()),
                offset: Some(1),
                size_in_bytes: 26,
                cardinality: 3,
            }
        );
        let file2 = result.get("file2.parquet").unwrap();
        assert_eq!(
            file2,
            &DeletionVectorDescriptor {
                storage_type: StorageType::UuidRelativePath,
                path_or_inline_dv: z85::encode(uuid.as_bytes()),
                offset: Some(35),
                size_in_bytes: 26,
                cardinality: 3,
            }
        );
    }

    #[test]
    fn test_write_one_deletion_vector() {
        let mut buffer = BytesMut::new();
        let result = write_deletion_vector(&mut buffer, &RoaringBitmap::from_iter(vec![1, 2, 3]))
            .expect("Failed to write deletion vector");
        assert_eq!(result.offset, 0);
        assert_eq!(result.size_in_bytes, 26);
        assert_eq!(result.cardinality, 3);
        assert_eq!(buffer.len(), 34);
    }

    #[test]
    fn test_read_deletion_vector_file() {}

    #[tokio::test]
    async fn test_delete_with_deletion_vectors() {
        let values: Arc<dyn Array> = Arc::new(arrow::array::StringArray::from(vec!["1", "2", "3"]));
        let batch = RecordBatch::try_from_iter(vec![("value", values)]).unwrap();
        let schema = batch.schema();

        // write some data
        let table = DeltaOps::new_in_memory()
            .write(vec![batch.clone()])
            .with_save_mode(SaveMode::Append)
            .await
            .unwrap();
        assert_eq!(table.version(), 0);
        assert_eq!(table.get_files_count(), 1);

        let snapshot = table.snapshot().expect("Failed to get snapshot");
        let mut adds = snapshot.file_actions_iter().expect("Failed to get file actions").collect::<Vec<_>>();
        assert_eq!(adds.len(), 1);
        let original_add_action = adds.pop().unwrap();

        let (table, metrics) = DeltaOps(table)
            .delete()
            .with_predicate(col("value").eq(lit(1)))
            .with_deletion_vectors(true)
            .await
            .unwrap();

        assert_eq!(table.version(), 1);
        assert_eq!(table.get_files_count(), 1);
        // TODO: Metrics are incorrect
        assert_eq!(metrics.num_added_files, 2);
        assert_eq!(metrics.num_removed_files, 1);
        assert_eq!(metrics.num_deleted_rows, 2);
        assert_eq!(metrics.num_copied_rows, 1);

        let snapshot = table.snapshot().expect("Failed to get snapshot");
        let mut adds = snapshot.file_actions_iter().expect("Failed to get file actions").collect::<Vec<_>>();
        assert_eq!(adds.len(), 1);
        let mut new_add_action = adds.pop().unwrap();
        let deletion_vector = new_add_action.deletion_vector.take().expect("No deletion vector");
        assert_eq!(original_add_action, new_add_action);
        assert_eq!(deletion_vector, DeletionVectorDescriptor {
            storage_type: StorageType::UuidRelativePath,
            path_or_inline_dv: deletion_vector.path_or_inline_dv.clone(),
            offset: Some(1),
            size_in_bytes: 22,
            cardinality: 1,
        });

        let root = Url::parse(&table.log_store().root_uri()).expect("Failed to parse URL");
        let deletion_vector_uri = deletion_vector.absolute_path(&root).expect("Failed to get absolute path").expect("No absolute path");
        let deletion_vector_path = Path::from(deletion_vector_uri.path());
        let deletion_vector = table.object_store().get(&deletion_vector_path).await.expect("Failed to get object");
        assert_eq!(deletion_vector.meta.size, 31);


        // let config = DeltaScanConfigBuilder::new()
        //     .build(table.snapshot().unwrap())
        //     .unwrap();
        // let provider = DeltaTableProvider::try_new(
        //     table.snapshot().unwrap().clone(),
        //     table.log_store(),
        //     config,
        // )
        // .unwrap();
        //
        // let ctx = SessionContext::new();
        // ctx.register_table("test", Arc::new(provider)).unwrap();
        // let state = ctx.state();
        // let df = ctx.sql("select value from test").await.unwrap();
        // let plan = df.create_physical_plan().await.unwrap();
        //
        // let mut stream = plan.execute(0, state.task_ctx()).unwrap();
        // let batches: Vec<RecordBatch> = stream.try_collect().await.expect("Failed to collect");
        // assert_eq!(1, batches.len());
        // let batch = &batches[0];
        // assert_eq!(2, batch.num_rows());
        //
        // let value = batch.column_by_name("value").unwrap().as_string::<i32>();
        // let values = value.iter().collect::<Vec<_>>();
        // assert_eq!(values, vec![Some("2"), Some("3")]);
    }

    #[tokio::test]
    async fn test_read_table_with_deletion_vectors() {

    }
}