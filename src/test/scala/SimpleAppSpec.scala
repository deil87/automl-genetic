
import org.scalacheck.{Gen, Prop, Properties}
import org.scalacheck.Prop.forAll
import Prop.BooleanOperators

object SimpleAppSpec extends Properties("String") {

  property("startsWith") = forAll { (a: String, b: String) =>
    (a+b).startsWith(a)
  }

  property("concatenate") = forAll { (a: String, b: String) =>
    (a != "" && b != "") ==> {
      (a + b).length > a.length && (a + b).length > b.length
    }
  }

  property("substring") = forAll { (a: String, b: String, c: String) =>
    (a+b+c).substring(a.length, a.length+b.length) == b
  }

  property("constains") = forAll { (a: String, b: String, c: String) =>
    (a+b+c).contains(b)
  }

  val testGen = Gen.choose(1, 100)
  val someInt: Option[Int] = testGen.sample

  trait Color
  case object Red extends Color
  case object Green extends Color

  trait Shape { def color: Color }
  case class Line(val color: Color) extends Shape
  case class Circle(val color: Color) extends Shape
  case class Box(val color: Color,
                 val boxed: Shape) extends Shape

  val genColor = Gen.oneOf(Red, Green)

  val genLine = for { color <- genColor } yield Line(color)
  val genCircle = for { color <- genColor } yield Circle(color)
  val genBox = for {
    color <- genColor
    shape <- genShape
  } yield Box(color, shape)

  val genShape: Gen[Shape] =
    Gen.oneOf(genLine, genCircle, genBox)

  println(genBox.sample)

}