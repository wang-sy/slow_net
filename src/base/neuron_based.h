//MIT License
//
//Copyright (c) 2020 wangsaiyu
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#ifndef SLOW_NET_NEURON_BASED_H
#define SLOW_NET_NEURON_BASED_H

#include <Eigen/Core>

/**
 * 神经元基类，一切神经元应当继承本基类
 * 在神经网络中，每一层(layer)都会包含若干个神经元(neuron)，在默认情况下，本项目仅会使用default_neuron神经元
 * 当然，你也可以对神经元进行自定义，为此，你只需要继承本类，并重写feed_forward、back_propagate两个函数即可
 * 如果你仍然没有看懂，可以参考src/default_neuron.h/cc 的实现来自己实现自己的神经元
 *
 * 神经元在预测(predict)过程中的主要任务是：
 * 接收来自上一层的输出信息，并且做出输出，该任务被称为前向传播(feed_forward)，对应下方的feed_forward函数
 * 神经元在训练过程中的主要任务是接受线性输出的成本梯度，并且对自身参数进行更新，
 * 该任务被称为反向传播(back_propagate)，对应下方的back_propagate函数
 * @author wangsaiyu@cqu.edu.cn
 */
class neuron_based {
protected:
    /**
     * 符号定义，凡是继承neuron_based神经元基类的类，都可以使用FloatVector类型的变量
     * FloatVector 表示一个行数为1的矩阵，该矩阵的列数不定，在使用过程中可以被视为向量
     */
    typedef Eigen::Matrix<float, 1, Eigen::Dynamic> FloatVector;

public:
    // 神经元节点的前向传播与后向传播算法，此处被定义为纯需函数，可供用户自定义，必须被重写
    virtual float feed_forward(FloatVector& _pre_layer_output) = 0; // 前向传播算法
    virtual void back_propagate(float _delta, FloatVector& _derivatives, FloatVector& _previous_result, float _learning_rate)= 0; // 后向传播算法
};


#endif //SLOW_NET_NEURON_BASED_H
