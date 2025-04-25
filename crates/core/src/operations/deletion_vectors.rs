use std::collections::HashMap;
use std::io;

use bytes::{BufMut, Bytes, BytesMut};
use uuid::Uuid;

use crate::errors::DeltaResult;
use crate::kernel::{DeletionVectorDescriptor, StorageType};
use roaring::RoaringTreemap;

const DELETION_VECTOR_MAGIC: [u8; 4] = 1681511377u32.to_be_bytes();
const DELETION_VECTOR_FILE_FORMAT_VERSION_1: u8 = 1;

mod size {
    pub const VERSION: usize = 1;
    pub const DATA_SIZE: usize = 4;
    pub const MAGIC: usize = 4;
    pub const CHECKSUM: usize = 4;
}

pub fn write_deletion_vectors_file<I>(
    uuid: &Uuid,
    deletion_vectors: I,
) -> DeltaResult<(HashMap<String, DeletionVectorDescriptor>, Bytes)>
where
    for<'a> &'a I: IntoIterator<Item = (&'a String, &'a RoaringTreemap)>,
    I: IntoIterator<Item = (String, RoaringTreemap)>,
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

#[derive(Debug, PartialEq)]
struct PartialDeletionVectorDescriptor {
    offset: i32,
    size_in_bytes: i32,
    cardinality: i64,
}

fn write_deletion_vector(
    buffer: &mut BytesMut,
    deletion_vector: &RoaringTreemap,
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

pub fn deserialize_deletion_vector(
    bytes: &[u8],
    descriptor: &DeletionVectorDescriptor,
) -> std::io::Result<RoaringTreemap> {
    let offset = descriptor.offset.unwrap_or_default() as usize + size::MAGIC + size::DATA_SIZE;
    let length = descriptor.size_in_bytes as usize;
    let bytes = &bytes[offset..offset + length];
    RoaringTreemap::deserialize_from(bytes)
}

#[cfg(test)]
mod tests {
    use crate::kernel::actions::DeletionVectorDescriptor;
    use crate::kernel::StorageType;
    use crate::operations::deletion_vectors::{
        deserialize_deletion_vector, PartialDeletionVectorDescriptor,
    };
    use bytes::BytesMut;
    use roaring::RoaringTreemap;
    use std::collections::BTreeMap;
    use uuid::Uuid;

    #[test]
    fn test_write_one_deletion_vector() {
        let mut buffer = BytesMut::new();
        let result =
            super::write_deletion_vector(&mut buffer, &RoaringTreemap::from_iter(vec![1, 2, 3]))
                .expect("Failed to write deletion vector");
        assert_eq!(result.offset, 0);
        assert_eq!(result.size_in_bytes, 38);
        assert_eq!(result.cardinality, 3);
        assert_eq!(buffer.len(), 46);
    }

    #[test]
    fn test_read_deletion_vector_file() {
        let mut buffer = BytesMut::new();
        let expected = RoaringTreemap::from_iter(vec![1, 2, 3]);
        let result = super::write_deletion_vector(&mut buffer, &expected)
            .expect("Failed to write deletion vector");
        assert_eq!(
            result,
            PartialDeletionVectorDescriptor {
                offset: 0,
                size_in_bytes: 38,
                cardinality: 3,
            }
        );
        let actual = super::deserialize_deletion_vector(
            &buffer.freeze(),
            &DeletionVectorDescriptor {
                storage_type: Default::default(),
                path_or_inline_dv: "".to_string(),
                offset: Some(0),
                size_in_bytes: 38,
                cardinality: 3,
            },
        )
        .expect("Failed to deserialize deletion vector");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_write_deletion_vectors_file() {
        use roaring::RoaringTreemap;
        let uuid = Uuid::from_u128(1);
        let vector1 = RoaringTreemap::from_iter(vec![1, 2, 3]);
        let vector2 = RoaringTreemap::from_iter(vec![4, 5, 6]);
        let (result, bytes) = super::write_deletion_vectors_file(
            &uuid,
            BTreeMap::from_iter([
                // BTreeMap instead of HashMap to have consistent key orders for non-flaky tests
                ("file1.parquet".to_string(), vector1.clone()),
                ("file2.parquet".to_string(), vector2.clone()),
            ]),
        )
        .expect("Failed to write deletion vectors file");
        assert_eq!(result.len(), 2);
        assert_eq!(bytes.len(), 93);

        let file1 = result.get("file1.parquet").unwrap();
        assert_eq!(
            file1,
            &DeletionVectorDescriptor {
                storage_type: StorageType::UuidRelativePath,
                path_or_inline_dv: z85::encode(uuid.as_bytes()),
                offset: Some(1),
                size_in_bytes: 38,
                cardinality: 3,
            }
        );
        assert_eq!(vector1, deserialize_deletion_vector(&bytes, file1).unwrap());

        let file2 = result.get("file2.parquet").unwrap();
        assert_eq!(
            file2,
            &DeletionVectorDescriptor {
                storage_type: StorageType::UuidRelativePath,
                path_or_inline_dv: z85::encode(uuid.as_bytes()),
                offset: Some(47),
                size_in_bytes: 38,
                cardinality: 3,
            }
        );
        assert_eq!(vector2, deserialize_deletion_vector(&bytes, file2).unwrap());
    }
}
