use crate::kernel::DeletionVectorDescriptor;
use crate::operations::deletion_vectors::deserialize_deletion_vector;
use crate::storage::ObjectStoreRef;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, BooleanArray, RecordBatch, StructArray};
use arrow_schema::{ArrowError, FieldRef, SchemaRef};
use bytes::Bytes;
use datafusion::datasource::physical_plan::{
    FileMeta, FileOpenFuture, FileOpener, FileScanConfig, FileSource, ParquetSource,
};
use datafusion::physical_expr::LexOrdering;
use datafusion_common::{DataFusionError, Statistics};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::DisplayFormatType;
use futures::future::Either;
use futures::stream::BoxStream;
use futures::{try_join, StreamExt, TryFutureExt};
use object_store::ObjectStore;
use roaring::RoaringTreemap;
use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;
use url::Url;

#[derive(Clone)]
pub struct DeletionVectorFileSource {
    inner: Arc<dyn FileSource>,
    object_store: ObjectStoreRef,
    row_number_column: FieldRef,
    is_row_deleted_column: FieldRef,
}

/// File source which adds the is_row_deleted_column based on the row_number_column and
/// DeletionVectorDescriptors from FileMeta::extension.
impl DeletionVectorFileSource {
    pub fn new(
        inner: ParquetSource,
        object_store: ObjectStoreRef,
        row_number_column: FieldRef,
        is_row_deleted_column: FieldRef,
    ) -> Self {
        Self {
            inner: Arc::new(inner),
            object_store,
            row_number_column,
            is_row_deleted_column,
        }
    }
}

impl FileSource for DeletionVectorFileSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Arc<dyn FileOpener> {
        let is_deleted_column_index = base_config
            .file_schema
            .index_of(self.is_row_deleted_column.name())
            .expect("is_deleted_column must be part of file schema");
        let base_config = cheaper_clone(base_config);
        let projection = base_config
            .projection
            .as_ref()
            .map(|projection| {
                projection
                    .iter()
                    .filter(|idx| **idx != is_deleted_column_index)
                    .cloned()
                    .collect()
            })
            .unwrap_or_else(|| {
                (0..base_config.file_schema.fields().len())
                    .filter(|idx| *idx != is_deleted_column_index)
                    .collect()
            });
        let base_config = base_config.with_projection(Some(projection));
        // At the moment, the only parts of base_config used by the Parquet opener is file schema and projection
        let inner = self
            .inner
            .create_file_opener(object_store, &base_config, partition);
        Arc::new(DeletionVectorFileOpener {
            inner,
            object_store: self.object_store.clone(),
            row_number_column: self.row_number_column.clone(),
            is_row_deleted_column: self.is_row_deleted_column.clone(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.inner = self.inner.with_batch_size(batch_size);
        Arc::new(conf)
    }

    fn with_schema(&self, _schema: SchemaRef) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn with_projection(&self, _config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(self.clone())
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        // This will contain statistics for the is_row_deleted_column, but it is ignored by the projection
        let mut conf = self.clone();
        conf.inner = self.inner.with_statistics(statistics);
        Arc::new(conf)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        self.inner.metrics()
    }

    fn statistics(&self) -> datafusion_common::Result<Statistics> {
        self.inner.statistics()
    }

    fn file_type(&self) -> &str {
        self.inner.file_type()
    }

    fn fmt_extra(&self, _t: DisplayFormatType, _f: &mut Formatter) -> std::fmt::Result {
        self.inner.fmt_extra(_t, _f)
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
        config: &FileScanConfig,
    ) -> datafusion_common::Result<Option<FileScanConfig>> {
        self.inner.repartitioned(
            target_partitions,
            repartition_file_min_size,
            output_ordering,
            config,
        )
    }
}

struct DeletionVectorFileOpener {
    inner: Arc<dyn FileOpener>,
    object_store: ObjectStoreRef,
    row_number_column: FieldRef,
    is_row_deleted_column: FieldRef,
}

impl FileOpener for DeletionVectorFileOpener {
    fn open(&self, file_meta: FileMeta) -> datafusion_common::Result<FileOpenFuture> {
        let deletion_vector_descriptor = file_meta
            .extensions
            .clone()
            .and_then(|ext| ext.downcast::<DeletionVectorDescriptor>().ok());
        let inner = self.inner.open(file_meta)?;
        let object_store = self.object_store.clone();
        let deletion_vector_future = match deletion_vector_descriptor {
            Some(descriptor) => {
                Either::Left(read_deletion_vector(object_store, descriptor).map_ok(Some))
            }
            None => Either::Right(futures::future::ok(None)),
        };
        let row_number_column = self.row_number_column.clone();
        let row_deleted_column = self.is_row_deleted_column.clone();
        Ok(Box::pin(async move {
            let (deletion_vector, stream) = try_join!(deletion_vector_future, inner)?;
            Ok(stream_with_delete_column(
                row_number_column,
                row_deleted_column,
                deletion_vector,
                stream,
            ))
        }))
    }
}

/// A "cheaper" clone of FileScanConfig. It avoids cloning the file_groups of FileScanConfig, as it
/// scales with the amount of data in the table, and it's unused by ParquetSource::create_file_opener.
/// The remaining fields scale with the size of the schema of the table, so even if it's annoying
/// that we clone this, it's less bad.
fn cheaper_clone(config: &FileScanConfig) -> FileScanConfig {
    let FileScanConfig {
        object_store_url,
        file_schema,
        // file_groups,
        constraints,
        statistics,
        projection,
        limit,
        table_partition_cols,
        output_ordering,
        file_compression_type,
        new_lines_in_values,
        file_source,
        ..
    } = config;
    FileScanConfig {
        object_store_url: object_store_url.clone(),
        file_schema: file_schema.clone(),
        file_groups: vec![], // Avoid cloning this, as it can be very large
        constraints: constraints.clone(),
        statistics: statistics.clone(),
        projection: projection.clone(),
        limit: *limit,
        table_partition_cols: table_partition_cols.clone(),
        output_ordering: output_ordering.clone(),
        file_compression_type: *file_compression_type,
        new_lines_in_values: *new_lines_in_values,
        file_source: file_source.clone(),
    }
}

async fn read_deletion_vector(
    object_store: ObjectStoreRef,
    deletion_vector_descriptor: Arc<DeletionVectorDescriptor>,
) -> datafusion_common::Result<RoaringTreemap> {
    let root = Url::parse("unused://").unwrap(); // TODO: Make static or something
    let path = deletion_vector_descriptor
        .absolute_path(&root)
        .map_err(|err| DataFusionError::External(Box::new(err)))?
        .map(|path| object_store::path::Path::parse(path.path()))
        .transpose()?;
    let bytes = if let Some(path) = path {
        object_store.get(&path).await?.bytes().await?
    } else {
        let decoded = z85::decode(&deletion_vector_descriptor.path_or_inline_dv)
            .map_err(|err| DataFusionError::External(Box::new(err)))?;
        Bytes::from(decoded)
    };
    deserialize_deletion_vector(&bytes, &deletion_vector_descriptor)
        .map_err(|err| DataFusionError::External(Box::new(err)))
}

fn stream_with_delete_column(
    row_number_column: FieldRef,
    is_row_deleted_column: FieldRef,
    deletion_vector: Option<RoaringTreemap>,
    stream: BoxStream<'static, Result<RecordBatch, ArrowError>>,
) -> BoxStream<'static, Result<RecordBatch, ArrowError>> {
    stream
        .map(move |batch| {
            println!("batch={:#?}", batch);
            let batch = batch?;
            let num_rows = batch.num_rows();
            let struct_array: StructArray = batch.into();

            let mut is_row_deleted_builder = BooleanArray::builder(num_rows);
            if let Some(ref deletion_vector) = deletion_vector {
                let row_number_column = struct_array
                    .column_by_name(row_number_column.name())
                    .and_then(|column| column.as_primitive_opt::<UInt64Type>())
                    .ok_or_else(|| {
                        ArrowError::SchemaError(format!(
                            "Row number column '{}' with type Uint64 not found in batch",
                            row_number_column.as_ref()
                        ))
                    })?;
                for row in row_number_column.iter() {
                    if let Some(row) = row {
                        is_row_deleted_builder.append_value(deletion_vector.contains(row));
                    } else {
                        is_row_deleted_builder.append_null();
                    }
                }
            } else {
                is_row_deleted_builder.append_n(num_rows, false);
            }
            let is_row_deleted: Arc<dyn Array> = Arc::new(is_row_deleted_builder.finish());

            let (fields, mut arrays, nulls) = struct_array.into_parts();
            let fields = fields
                .into_iter()
                .cloned()
                .chain(std::iter::once(is_row_deleted_column.clone()))
                .collect();
            arrays.push(is_row_deleted);

            Ok(StructArray::try_new(fields, arrays, nulls)?.into())
        })
        .boxed()
}
