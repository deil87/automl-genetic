package utils

import com.typesafe.scalalogging.LazyLogging
import org.scalameter.{Key, Warmer, _}

import scala.util.control.NonFatal

object BenchmarkHelper extends LazyLogging{

  def time[R](marker: String)(block: => R): R = {
    logger.debug(s"Benchmark $marker started.")
    val t0 = System.currentTimeMillis()
    var result: Any = null
    try {
      result = block
    } catch {
      case NonFatal(ex) =>
        logger.debug(s"$marker# Failure: ${ex.getMessage}")
    } finally {
      val t1 = System.currentTimeMillis()
      logger.debug(s"Benchmark. $marker took: " + (t1 - t0) + "ms")
    }
    result.asInstanceOf[R]
  }


  //SCALAMETER

  /* val amount = standardConfig measure{
      val tmp =  (1 to 7).toList.map{ i =>
        Future { ...
    }
    println("Minimum value:" + minValue)*/
  /*val standardConfig = config(
    Key.exec.minWarmupRuns -> 20,
    Key.exec.maxWarmupRuns -> 40,
    Key.exec.benchRuns -> 1000000,
    Key.verbose -> true
  ) withWarmer(new Warmer.Default)*/
}
