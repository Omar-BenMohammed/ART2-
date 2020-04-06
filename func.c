#include"func.h"

/********************************************************************************************/

double** init_ltm(int l, int c, double val_init)
{
    
    double** matrice = NULL ;
	matrice = malloc(l * sizeof(double*));

    for (int i = 0; i < l; i++)
    {
        matrice[i] = malloc(c * sizeof(double));

        for (int j = 0; j < c; j++)
        { 
            matrice[i][j] = val_init ; 
        }
    }
    return matrice ; 
}

/********************************************************************************************/

network_t* init_net(int size_vec,int nb_class, double* vec_init)
{

    double val_bu = 0.0 ; 
    network_t* network = malloc(sizeof(network_t)); 
    network->winner = 1 ; 
    network->size_vec = size_vec;
    network->nb_class = nb_class;
    network->a = vec_init[A];
    network->b = vec_init[B];
    network->c = vec_init[C];
    network->d = vec_init[D]; 
    network->e = vec_init[E];
    network->sig = vec_init[SIG];
    network->vigilance = vec_init[VIGILANCE];
    network->reset = 1;

    network->cpt = 0;
	network->w = calloc(network->size_vec, sizeof(double));
	network->x = calloc(network->size_vec, sizeof(double));
	network->v = calloc(network->size_vec, sizeof(double));
	network->u = calloc(network->size_vec, sizeof(double));
	network->p = calloc(network->size_vec, sizeof(double));
	network->q = calloc(network->size_vec, sizeof(double));
	network->r = calloc(network->size_vec, sizeof(double));
	network->f2 = calloc(network->nb_class, sizeof(double));

    network->top_down =  init_ltm(network->size_vec, network->nb_class,0.0) ; 
    
    val_bu = 0.5/((1-network->d)*sqrt((double)network->size_vec)) ; 
    network->bottom_up =  init_ltm(network->nb_class,network->size_vec,val_bu) ; 

    printf("Matrice Top Down \n"); 
    print_matrice(network->top_down, network->size_vec, network->nb_class);
    printf("\n \n \n Matrice Bottom Up \n");
    print_matrice(network->bottom_up, network->nb_class, network->size_vec);
    printf("\n \n \n");

    return network ; 
}



/********************************************************************************************/

void calc_w(network_t* network, double *input)
{
    for (int i = 0; i < network->size_vec; i++)
    {
        network->w[i] = input[i] + (network->a * network->u[i]);
    }    
}
/********************************************************************************************/
double norm(double* vec, int size)
{
    double somme = 0.0 ;
    for (int i = 0; i < size; i++)
    {
        somme += vec[i]*vec[i]; 
    }

    return sqrt(somme) ; 
}
/********************************************************************************************/
void norm_vec(double* dst, double* src,double val, int size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = src[i] / val ;
    }
    
}
/********************************************************************************************/
double sigmoid(double val, double seuil)
{
    if (val > seuil){
        return val ; 
    }else
    {
        return 0.0 ; 
    } 
}
/********************************************************************************************/
void calc_v(network_t *network)
{
    for (int i = 0; i < network->size_vec; i++)
    {
        network->v[i] = sigmoid(network->x[i], network->sig) + network->b * sigmoid(network->q[i], network->sig);
    }
    
}
/********************************************************************************************/
void calc_p(network_t* network)
{

    for (int i = 0; i < network->size_vec ; i++)
    {
       
        network->p[i] = network->u[i] + (network->b * network->top_down[i][network->winner]);  
    
    }   
}


/********************************************************************************************/
void init_f1(network_t *network, double* input)
{
    int cpt = 0 ;  
    double tmp_norm = 0.0 ; 

    while (cpt < 3 )
    {

        //calc U
        tmp_norm = norm(network->v, network->size_vec) + network->e; 
        norm_vec(network->u, network->v,tmp_norm, network->size_vec);

        calc_w(network, input);

        calc_p(network); 

        //calc Q
        tmp_norm = norm(network->p, network->size_vec)+ network->e; 
        norm_vec(network->q, network->p,tmp_norm, network->size_vec);
       

        //clac X
        tmp_norm = norm(network->w, network->size_vec)+ network->e; 
        norm_vec(network->x, network->w,tmp_norm, network->size_vec); 

        calc_v(network);

        cpt++ ;  
    } 
     
}

/********************************************************************************************/

void calc_r(network_t* network)
{
    double norm_u = norm(network->u, network->size_vec);
    double norm_p = norm(network->p, network->size_vec);
    for (size_t i = 0; i < network->size_vec; i++)
    {
        network->r[i] = (network->u[i] + network->c * network->p[i]) / (network->e + norm_u + (network->c * norm_p)) ;    
    }
    
}

/********************************************************************************************/

void init_f2(network_t *network)
{
    for (int j = 0; j < network->nb_class; j++)
    {
        for (int i = 0; i < network->size_vec; i++)
        {
            network->f2[j] += network->bottom_up[j][i] * network->p[i]; 
        }
    }
    printf("F2 = "); 
    print_vec(network->f2, network->nb_class);
}

/********************************************************************************************/

int get_winner(double *vec, int size)
{
    int winner = 0 ; 
    for (int i = 1; i < size; i++)
    {
        if (vec[i]>vec[winner])
        {
            winner = i ; 
        }
        
    }
    return winner ; 
}

/********************************************************************************************/

void or_sys(network_t *network, double *input)
{
    
    while(network->reset && network->cpt < network->nb_class)
    {
        network->winner = get_winner(network->f2, network->nb_class);
        printf("winner : %d \n \n",network->winner); 
        //calc U
        double tmp_norm = norm(network->v, network->size_vec); 
        norm_vec(network->u, network->v,tmp_norm, network->size_vec);
        
        calc_p(network); 
        calc_r(network); 

        double norm_r = norm(network->r,network->size_vec); 

        if(norm_r <= network->vigilance){

            network->f2[network->winner] = -1 ; 
            network->reset = 1 ;

        }else{
            calc_w(network,input); 

            //calc X
            tmp_norm = norm(network->w, network->size_vec); 
            norm_vec(network->x, network->w,tmp_norm, network->size_vec); 

            calc_v(network); 

            network->reset = 0 ; 
        }
        network->cpt += 1 ; 
    }
}

/********************************************************************************************/

void learning(network_t *network)
{
    for (int i = 0; i < network->size_vec; i++)
    {
        network->top_down[i][network->winner] = network->u[i]/ (1-network->d);
        network->bottom_up[network->winner][i] = network->u[i]/ (1-network->d);
    }
    printf("Matrice Top Down \n"); 
    print_matrice(network->top_down, network->size_vec, network->nb_class);
    printf("\n \n \n Matrice Bottom Up \n");
    print_matrice(network->bottom_up, network->nb_class, network->size_vec);
    printf("\n \n \n");

    for (int i = 0; i < network->nb_class; i++)
    {
        network->f2[i] = -1 ; 
    }
    network->reset = 1 ;
    network->cpt = 0 ; 
}




















/********************************************************************************************/


void print_vec(double* v, int count)
{
	for (int i = 0; i < count; i++)
	{
		printf("%f ",v[i]); 
	}
	printf("\n");
}


void print_net(network_t *network)
{
	printf("w = ");
	print_vec(network->w, network->size_vec); 
	printf("x = ");
	print_vec(network->x, network->size_vec); 
	printf("v = ");
	print_vec(network->v, network->size_vec); 
	printf("u = ");
	print_vec(network->u, network->size_vec); 
	printf("p = ");
	print_vec(network->p, network->size_vec); 
	printf("q = ");
	print_vec(network->q, network->size_vec); 

}


void print_matrice(double** matrice, int l, int c)
{

    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < c; j++)
        {
            printf("%f ",matrice[i][j]);
        }
        printf("\n"); 
    }
    
}





void free_network(network_t *network)
{
    for (int im = 0; im < network->size_vec; ++im)
	{
		free(network->top_down[im]);
	}

	for (int i = 0; i < network->nb_class; ++i)
	{
		free(network->bottom_up[i]);
	}
   
	free(network->f2);
    free(network->r);
    free(network->q);
    free(network->p);
    free(network->u);
    free(network->v);
    free(network->x);
    free(network->w);
	free(network); 
}