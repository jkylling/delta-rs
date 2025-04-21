```scala
package org.apache.spark.sql.delta

import org.apache.spark.sql.QueryTest
import org.apache.spark.sql.delta.sources.DeltaSQLConf
import org.apache.spark.sql.delta.test.{DeltaSQLCommandTest, DeltaSQLTestUtils}
import org.apache.spark.sql.test.SharedSparkSession

class DeltaRsSuite extends QueryTest
  with SharedSparkSession
  with DeltaColumnMappingTestUtils
  with DeltaSQLTestUtils
  with DeltaSQLCommandTest {

  val testPath = "~/delta-rs/crates/core/tests/data/spark/deletion-vectors-2"

  import testImplicits._

  test("handle partition filters and data filters") {
    withSQLConf(
      DeltaConfigs.ENABLE_DELETION_VECTORS_CREATION.defaultTablePropertyKey -> "true",
      DeltaSQLConf.DELETE_USE_PERSISTENT_DELETION_VECTORS.key -> true.toString) {

      spark.range(1, 4)
        .map(_.toInt)
        .withColumn("value", $"value")
        .write
        .format("delta")
        .mode("append")
        .save(testPath)

      spark.range(1, 4)
        .map(_.toInt)
        .withColumn("value", $"value")
        .write
        .format("delta")
        .mode("append")
        .save(testPath)

      val deltaTable: io.delta.tables.DeltaTable =
        DeltaTestUtils.getDeltaTableForIdentifierOrPath(
          spark,
          DeltaTestUtils.TableIdentifierOrPath.Path(testPath, None))

      deltaTable.delete("value = 2")
    }
  }
}
```