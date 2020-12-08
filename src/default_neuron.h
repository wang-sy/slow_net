//
// Created by wangsy on 2020/12/3.
//

#ifndef SLOW_NET_DEFAULT_NEURON_H
#define SLOW_NET_DEFAULT_NEURON_H

#include "base/neuron_based.h"

// 默认神经元，继承自base/neuron_based
class default_neuron: neuron_based{
private:
    FloatVector _w; // 神经元中的w参数，表示上一层传入的每一个参数的权重
public:
    // 对neuron_based中feed_forward方法的实现，完成神经网络中神经元的前向传播算法
    float feed_forward(FloatVector& _pre_layer_output) override; // 前向传播算法
    void back_propagate(float _delta, FloatVector& _derivatives, FloatVector& _previous_result, float _learning_rate) override;
};


#endif //SLOW_NET_DEFAULT_NEURON_H
