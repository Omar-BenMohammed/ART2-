#include"func.h"

int main(int argc, char const *argv[])
{
    double vec_init[SIZE_VEC] = {10.0,10.0,0.1,0.9,0.000000000001,0.2,0.9}; //A,B,C,D,E,SIG,VIGILANCE,SIZE_VEC
    double input[5] = {0.2,0.7,0.1,0.5,0.4}; 

    network_t* network = init_net(5,6,vec_init);


    init_f1(network,input); 
    init_f2(network);
    or_sys(network,input) ; 
    learning(network);
   
    
    
    free_network(network);

    
    return 0;
}
