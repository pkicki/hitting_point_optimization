#ifndef OPTIMIZER_DATA_H
#define OPTIMIZER_DATA_H
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>

using namespace std;

struct OptimizerData {
    Eigen::Vector3d hitPoint;
    Eigen::Vector3d hitDirection;
    Eigen::VectorXd q0;
    double epsilon;
    pinocchio::Model pinoModel;
	pinocchio::Data pinoData;
	int pinoFrameId;

    OptimizerData(string robot_path) {
       pinocchio::urdf::buildModel(robot_path, pinoModel);
       pinoData = pinocchio::Data(pinoModel);
       pinoFrameId = pinoModel.getFrameId("F_striker_tip");
       epsilon = sqrt(numeric_limits<double>::epsilon());
   }

};
#endif
