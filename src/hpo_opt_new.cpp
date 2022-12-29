#include <iostream>
#include "hpo_opt_new.h"
#include "optimizer_data.h"
#include <chrono>

using namespace std;
using namespace nlopt;
using namespace Eigen;

HittingPointOptimizerSingle::HittingPointOptimizerSingle(string robot_path): optData(OptimizerData(robot_path)) {

    optimizer = opt(LD_SLSQP, optData.pinoModel.nq);
    optimizer.set_max_objective(objective, &optData);


    double tol = 1e-5;
    optimizer.add_equality_constraint(this->equalityConstraint, &optData, tol);

    std::vector<double> qLower(optData.pinoModel.lowerPositionLimit.data(),
	                           optData.pinoModel.lowerPositionLimit.data() +
	                           optData.pinoModel.lowerPositionLimit.cols() *
	                           optData.pinoModel.lowerPositionLimit.rows());
	optimizer.set_lower_bounds(qLower);

	std::vector<double> qUpper(optData.pinoModel.upperPositionLimit.data(),
	                           optData.pinoModel.upperPositionLimit.data() +
	                           optData.pinoModel.upperPositionLimit.cols() *
	                           optData.pinoModel.upperPositionLimit.rows());

    optimizer.set_upper_bounds(qUpper);


    optimizer.set_ftol_abs(1e-8);
    optimizer.set_xtol_abs(1e-10);

    simplexModel.setLogLevel(0);
}

HittingPointOptimizerSingle::~HittingPointOptimizerSingle(){

}

bool HittingPointOptimizerSingle::solve(const Eigen::Vector3d& hitPoint, const Eigen::Vector3d& hitDirection,
                                     JointArrayType& qInOut, double &velMagMax, Eigen::Matrix<double, 9, 1>& qDir, double& final_f) {
    optData.hitPoint = hitPoint;
    optData.hitDirection = hitDirection;
    optData.q0 = qInOut;

    if (!getInitPoint(qInOut)){
        return false;
    }

    std::vector<double> qCur(qInOut.data(), qInOut.data() + qInOut.rows() * qInOut.cols());
    std::vector<double> grad(9);

    double opt_fun;
    auto result = optimizer.optimize(qCur, opt_fun);

    if (result < 0){
        return false;
    }

    if (h(qCur, &optData) < 1e-4) {
        qInOut = JointArrayType::Map(qCur.data(), qCur.size());
        velMagMax = getMaxVelocityLP(qInOut, qDir);
        return true;
    } else {
        cout << "The position error is : " << h(qCur, &optData) << " bigger than 1e-4" << endl;
        return false;
    }
}

double HittingPointOptimizerSingle::objective(const std::vector<double> &x, std::vector<double> &grad, void *f_data) {
    auto optData = (OptimizerData *) f_data;
    if (!grad.empty()){
        numerical_grad(f, x, optData, grad);
    }
    return f(x, optData);
}

double HittingPointOptimizerSingle::equalityConstraint(const vector<double> &x, vector<double> &grad, void *f_data) {
    auto optData = (OptimizerData *) f_data;
    if (!grad.empty()){
        numerical_grad(h, x, optData, grad);
    }
    return h(x, optData);
}

double HittingPointOptimizerSingle::inequalityConstraint(const vector<double> &x, vector<double> &grad, void *f_data) {
    auto optData = (OptimizerData *) f_data;
    if (!grad.empty()){
        numerical_grad(g, x, optData, grad);
    }
    return g(x, optData);
}

double HittingPointOptimizerSingle::f(const vector<double> &x, OptimizerData *data) {
    Eigen::Map<const JointArrayType> qCur(x.data(), x.size());
	pinocchio::Data::Matrix6x jacobian(6, data->pinoModel.nv);
	pinocchio::computeFrameJacobian(data->pinoModel, data->pinoData, qCur,
	                                data->pinoFrameId, pinocchio::LOCAL_WORLD_ALIGNED,
	                                jacobian);
    auto vec = data->hitDirection.transpose() * jacobian.topRows(3);
    auto a = (data->q0.topRows(6) - qCur.topRows(6));
    auto dist = a.cwiseProduct(data->pinoModel.velocityLimit.topRows(6).cwiseInverse());
    auto ret = vec.squaredNorm() - 1e0 * dist.squaredNorm();
    return ret;
}

double HittingPointOptimizerSingle::h(const vector<double> &x, OptimizerData *data) {
    JointArrayType qCur = JointArrayType::Map(x.data(), x.size());

	Eigen::Vector3d xCur;
	pinocchio::forwardKinematics(data->pinoModel, data->pinoData, qCur);
	pinocchio::updateFramePlacements(data->pinoModel, data->pinoData);
	xCur = data->pinoData.oMf[data->pinoFrameId].translation();

	auto distance = (xCur - data->hitPoint).norm();
    return distance;
}


double HittingPointOptimizerSingle::g(const vector<double> &x, OptimizerData *data) {
    JointArrayType qCur = JointArrayType::Map(x.data(), x.size());
	pinocchio::forwardKinematics(data->pinoModel, data->pinoData, qCur);
	pinocchio::updateFramePlacements(data->pinoModel, data->pinoData);
	int elbow_index = data->pinoModel.getFrameId("F_joint_4");
	return 0.2 - data->pinoData.oMf[elbow_index].translation()[2];
}

void HittingPointOptimizerSingle::numerical_grad(HittingPointOptimizerSingle::functype function, const vector<double> &x,
                                           OptimizerData *data, vector<double> &grad) {
    vector<double> x_pos, x_neg;
    for (size_t i = 0; i < x.size(); ++i) {
        x_pos = x;
        x_neg = x;
        x_pos[i] += data->epsilon;
        x_neg[i] -= data->epsilon;
        grad[i] = (function(x_pos, data) - function(x_neg, data))/ (2 * data->epsilon);
    }
}

bool HittingPointOptimizerSingle::getInitPoint(JointArrayType &qInOut) {
    if (!inverseKinematicsPosition(optData, optData.hitPoint, qInOut, qInOut, 1e-4, 200)) {
		qInOut = qInOut.cwiseMax(optData.pinoModel.lowerPositionLimit).cwiseMin(
				optData.pinoModel.upperPositionLimit);
		return false;
	}
    return true;
}

double HittingPointOptimizerSingle::getMaxVelocity(const JointArrayType &q, Eigen::Matrix<double, 9, 1>& qDir) {
    pinocchio::Data::Matrix6x jacobian(6, optData.pinoModel.nv);
    pinocchio::computeFrameJacobian(optData.pinoModel, optData.pinoData, q,
                                    optData.pinoFrameId, pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
    auto jac = jacobian.topRows(3);
    auto jacInv = jac.transpose() * (jac * jac.transpose()).inverse();
    auto qRef = jacInv * optData.hitDirection;
    double max_scale = 0.;
    for (auto i = 0; i < 6; ++i) {
        max_scale = max(max_scale, abs(qRef[i]) / (0.8 * optData.pinoModel.velocityLimit.data()[i]));
    }
    auto ratio = 1. / max_scale;
    qDir = qRef * ratio;
    return ratio;
}

double HittingPointOptimizerSingle::getMaxVelocityLP(const JointArrayType &q, Eigen::Matrix<double, 9, 1>& qDir) {
	pinocchio::Data::Matrix6x jacobian(6, optData.pinoModel.nv);
	pinocchio::computeFrameJacobian(optData.pinoModel, optData.pinoData, q,
	                                optData.pinoFrameId, pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
	auto jac = jacobian.topRows(3);

	MatrixXd orthogonalComplement;
	getNullSpace(optData.hitDirection.transpose(), orthogonalComplement);

	MatrixXd OCJac = (orthogonalComplement.transpose() * jac);
	MatrixXd OCJacNullSpace;
	getNullSpace(OCJac, OCJacNullSpace);

	// Get Objective
	MatrixXd objective = -optData.hitDirection.transpose() * jac * OCJacNullSpace;

	// Get Bounds for Primal Variables
	int nRows = OCJacNullSpace.rows();
	int nCols = OCJacNullSpace.cols();
	VectorXd columnLower(nCols);
	VectorXd columnUpper(nCols);
	columnLower = -columnLower.setOnes() * COIN_DBL_MAX;
	columnUpper = columnUpper.setOnes() * COIN_DBL_MAX;

	// Get Constraint Matrix
	CoinPackedMatrix matrix;
	matrix.setDimensions(nRows, nCols);
	for (int i = 0; i < OCJacNullSpace.rows(); ++i) {
		for (int j = 0; j < OCJacNullSpace.cols(); ++j) {
			matrix.modifyCoefficient(i, j, OCJacNullSpace(i, j));
		}
	}

	// Generate Problem
    optData.pinoModel.velocityLimit *= 0.8;
	Eigen::VectorXd velLowerLimit = -optData.pinoModel.velocityLimit;
	simplexModel.loadProblem(matrix, columnLower.data(), columnUpper.data(), objective.data(),
	                         velLowerLimit.data(), optData.pinoModel.velocityLimit.data());
	// Get Solution
	simplexModel.initialSolve();
	simplexModel.primal(0);

    int numberRows = simplexModel.numberRows();
    double * rowPrimal = simplexModel.primalRowSolution();
    for (int i=0;i<numberRows;i++) {
        qDir[i] = rowPrimal[i];
    }
	return -simplexModel.getObjValue();
}


boost::python::list optimize(string robot_path, double x, double y, double z, double cth, double sth, boost::python::list q0) {
//boost::python::list optimize(double x, double y, double z, double cth, double sth) {
    auto start = chrono::high_resolution_clock::now();
    HittingPointOptimizerSingle optimizer(robot_path);

    Eigen::Vector3d hitPos, hitDir;
    Eigen::Matrix<double, 9, 1> qDir;
    JointArrayType qInOut(9);
    for (auto i = 0; i < 9; i++) {
        qInOut[i] = boost::python::extract<double>(q0[i]);
    }
    double velMag = 0.;
    double final_f = 0.;

    hitPos.x() = x;
    hitPos.y() = y;
    hitPos.z() = z;
    hitDir.x() = cth;
    hitDir.y() = sth;
    hitDir.z() = 0.;
    hitDir.normalize();

    bool ret = optimizer.solve(hitPos, hitDir, qInOut, velMag, qDir, final_f);
    auto finish = chrono::high_resolution_clock::now();

    boost::python::list list;
    if (ret) {
        cout << qInOut << endl;
        cout << qDir << endl;
        cout << velMag << endl;
    } else {
        return list;
    }
    for (auto i=0; i < 9; i++) {
        list.append(qInOut[i]);
    }
    for (auto i=0; i < 9; i++) {
        list.append(qDir[i]);
    }
    list.append(velMag);
    list.append(final_f);

    cout << "TIME[ms]: " << chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() / 1.e6;
    return list;
}

BOOST_PYTHON_MODULE(hpo_opt_new)
        {
                using namespace boost::python;
                def("optimize", optimize);
        }

int main(int argc, char *argv[]) {
    return 0;
}


