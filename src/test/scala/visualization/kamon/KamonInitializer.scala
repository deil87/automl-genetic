package visualization.kamon

import kamon.Kamon

object KamonInitializer {

  import kamon.prometheus.PrometheusReporter
  Kamon.addReporter(new PrometheusReporter())
}
