package utils

import com.automl.PaddedLogging
import com.typesafe.scalalogging.LazyLogging
import org.scalameter.{Key, Warmer, _}

import scala.util.control.NonFatal

case class BenchmarkHelper(marker: String, logPaddingSize: Int) extends PaddedLogging {

  def apply[R](block: this.type => R): R = BenchmarkHelper.time(marker)(block(this))(logPaddingSize)

}

object BenchmarkHelper extends PaddedLogging {


  override def logPaddingSize: Int = 0

  def time[R](marker: String)(block: => R)(implicit logPaddingSize: Int): R = {

    debugImpl(s"Benchmark << $marker >> started.")
    val t0 = System.currentTimeMillis()
    var result: Any = null
    try {
      result = block
    } catch {
      case NonFatal(ex) =>
        debugImpl(s"$marker#\n\n !!!!!!! Failure: ${ex.getMessage}")
        throw ex
    } finally {
      val t1 = System.currentTimeMillis()
      val endMsg = s"Benchmark << $marker >> took: " + (t1 - t0) + "ms"
      debugImpl(endMsg)
      println(endMsg)

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
