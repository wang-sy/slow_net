//
// Created by wangsy on 2020/12/3.
//

#include "default_neuron.h"

/**
 * 神经网络中，单个神经元的前向传播算法
 * @param _pre_layer_output 输入，神经网络中上一层的输出，一个Eigen::Matrix类型的一维矩阵
 * @return 返回一个整数，表示本神经元的输出
 *
 * <p>约定： 输入的上一层的输出是一个(1, n)的矩阵 </p>
 * <p>方法： 神经元中的_w参数矩阵也是一个(1, n)的矩阵，故计算： </p>
 * <p>_pre_layer_output * this->_w.T + this->_b 就可以得到最终的答案 </p>
 */
float default_neuron::feed_forward(FloatVector &_pre_layer_output) {
    return _pre_layer_output * this->_w.transpose();
}

/**
 * 神经网络中，单个神经元的反向传播算法
 * @param _delta 输入，本层的输出对最后的误差的导数
 *
 * <ul>
 *  <li>更改_derivatives，将上一层的所有值相对最终误差的导数写入</li>
 *  <li>更改神经元自身参数</li>
 * </ul>
 */
void default_neuron::back_propagate(float _delta, FloatVector& _derivatives, FloatVector& _previous_result, float _learning_rate) {
    this->_w += _learning_rate * _delta;
}
