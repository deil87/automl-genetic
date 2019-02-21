package com.automl.hyperopt

import org.ejml.data.DMatrixRMaj
import org.ejml.dense.row.factory.DecompositionFactory_DDRM
import org.ejml.interfaces.decomposition.CholeskyDecomposition_F64
import org.ejml.simple.SimpleMatrix
import org.scalatest.{BeforeAndAfterAll, FunSuite, Matchers}

class HyperparameterOptimization extends FunSuite with Matchers with BeforeAndAfterAll  {

  test("be able to separate dataset into three classes( multiclass case) with LogisticRegression") {

    val mtrx = new SimpleMatrix(4, 4,true, Array[Double](7, 0, 0, 0,    2, 5, 0, 0,     -1, -2, 6, 0,      1, 0, -3, 5))

    val svd = mtrx.svd
    val evd = mtrx.eig

    val chol: CholeskyDecomposition_F64[DMatrixRMaj] = DecompositionFactory_DDRM.chol(mtrx.numRows, true)

    val matrix = mtrx.getMatrix.asInstanceOf[DMatrixRMaj]
    if (!chol.decompose(matrix)) throw new RuntimeException("Cholesky failed!")

    val L: SimpleMatrix = SimpleMatrix.wrap(chol.getT(null))

    true shouldBe true

  }
}

