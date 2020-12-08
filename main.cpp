#include <iostream>
#include <Eigen/Core>
using namespace std;

void output_eigenMatrix(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& _matrix){
    std::cout << _matrix << std::endl;
}

int main() {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> temp;
    Eigen::Matrix<float,  Eigen::Dynamic, Eigen::Dynamic> temp2;
    temp.resize(1, 5);

    temp2.resize(5, 1);
    temp << 1,2,3,4,5;
    cout << temp << endl;
    temp2 << 1,2,3,4,5;
    output_eigenMatrix(temp);
    output_eigenMatrix(temp2);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
