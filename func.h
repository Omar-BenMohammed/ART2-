#ifndef FUNC_H
#define FUNC_H
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>


enum{
    A = 0,
    B,C,D,E,SIG,VIGILANCE,SIZE_VEC
};

typedef struct network network_t;
struct network
{
    int size_vec,nb_class,winner,reset,cpt;
    double a,b,c,d,e,sig,vigilance; 
    double *w,*x,*v,*u,*p,*q,*r,*f2;
    double **top_down, **bottom_up;
};

network_t* init_net(int size_vec,int nb_class, double* vec_init);
double** init_ltm(int l, int c, double val_init);
void calc_w(network_t *network , double *input);
double norm(double* vec, int size);
void norm_vec(double* dst, double* src,double val, int size);
double sigmoid(double val, double seuil); 
void calc_v(network_t *network);
void calc_p(network_t* network);
void calc_r(network_t* network);
void init_f1(network_t *network, double *input);
void init_f2(network_t *network); 
int get_winner(double *vec, int size); 
void or_sys(network_t *network, double *input); 
void learning(network_t *network);

void print_vec(double* v, int count);
void print_net(network_t *network);
void print_matrice(double** matrice, int l, int c); 

void free_network(network_t *network); 

#endif