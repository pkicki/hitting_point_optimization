#ifndef SRC_HITTING_POINT_OPTIMIZER_H
#define SRC_HITTING_POINT_OPTIMIZER_H

#include <nlopt.hpp>
#include "utils.h"
#include <coin/ClpSimplex.hpp>
#include <coin/ClpSolve.hpp>
#include <boost/python.hpp>
#include "pinocchio/algorithm/rnea.hpp"


typedef Eigen::VectorXd JointArrayType;

class HittingPointOptimizerSingle {
public:
    HittingPointOptimizerSingle(string robot_path);

    ~HittingPointOptimizerSingle();

    bool solve(const Eigen::Vector3d &hitPoint, const Eigen::Vector3d &hitDirection,
               JointArrayType &qInOut, double &velMagMax,
               Eigen::Matrix<double, 9, 1>& qDir, double& final_f);

private:
    typedef double (*functype)(const std::vector<double> &x, OptimizerData *data);

    static double objective(const std::vector<double> &x, std::vector<double> &grad, void *f_data);

    static double equalityConstraint(const std::vector<double> &x, std::vector<double> &grad, void *data);

    static double inequalityConstraint(const std::vector<double> &x, std::vector<double> &grad, void *data);

    static double f(const std::vector<double> &x, OptimizerData *data);

    static double h(const std::vector<double> &x, OptimizerData *data);

    static double g(const std::vector<double> &x, OptimizerData *data);

    static void numerical_grad(functype function, const std::vector<double> &x, OptimizerData *data,
                               std::vector<double> &grad);

    bool getInitPoint(JointArrayType &qInOut);

    double getMaxVelocityLP(const JointArrayType &q, Eigen::Matrix<double, 9, 1> &qDir);
    double getMaxVelocity(const JointArrayType &q, Eigen::Matrix<double, 9, 1> &qDir);

public:
    static int it;


private:
    nlopt::opt optimizer;

    OptimizerData optData;
    ClpSimplex simplexModel;

};

#endif //SRC_HITTING_POINT_OPTIMIZER_H
