#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <algorithm>

using namespace std;

// Constantes

#define ClassNum 3
#define FeatureVectorSize 4
#define SampleNum 150
#define k 10
#define veck 3
// numero de segmentos e iteraciones para k fold cross validation
// definicion de Estructura
typedef struct iris_sample_type {
	char label[16];
	int id;
	int fv[FeatureVectorSize];
} IRISSample;

//Prototipos de funciones
static void load_iris_data();
//La función grupos trabaja bien cuando todas las clases tienen un
//mismo numero de instancias y cuando ese número de instancias es divisible
//en entero entre k.
static void grupos(int);
static int distan(int *,int);
static float pitagorica(int [4],int [4]);
static int vecinos(float [][2], int);
static int maximo(int [3]);

//Variables
static IRISSample iris[SampleNum];
//int randome[50];
int conjfund[ClassNum][45]; //indices de matrizmixta
int grupok[ClassNum][5];  //conjunto k en cada iteracion;
float percrecu[10];
float percprue[10];

//******************main*************************

int main(int argc, char *argv[]) {
    float factordeolvido=0;
    float factoreficiencia=0;
   	load_iris_data();
    printf("\n\n ***** K fold cross validation con knn ******\n\n");
    for (int n=0;n<k;n++)
        {
         grupos(n);
         percrecu[n]=distan((int*)conjfund,135);
         percrecu[n]=((float)(135-percrecu[n])/135)*100;
         percprue[n]=distan((int*)grupok,15);
         percprue[n]=(float)percprue[n]/15*100;
         factordeolvido+=*(percrecu+n);
         factoreficiencia+=*(percprue+n);
        }
    printf("Fase de recuperacion\tFase de prueba\n");
    for (int n=0;n<k;n++)
        {
        printf("%.2f % \t\t\t%.2f % \n",*(percrecu+n),*(percprue+n));
        }
    cout<<"\nfactor de olvido : "<<factordeolvido/10<<"%\t"<<"factor de eficiencia :"<<factoreficiencia/10<<"%\n";

	return 0;
}

// *********Carga los datos desde archivo uci.txt**********
void load_iris_data() {
	FILE *fp;
	char label[16];
	int id;
	int v0, v1, v2, v3;
	int n, i;

	if ((fp = fopen("uci.txt", "r")) == NULL) {
		cout << "No se puede abrir archivo" << endl;
		exit(1);
	}

	n = 0; //La función feof retorna un valor distinto a cero si y sólo si el indicador de final de fichero está activado para stream.
	while (!feof(fp)) {
		fscanf(fp, "%d%d%d%d%s%d\n", &v0, &v1, &v2, &v3, label, &id); // strcpy: Asigna una expresión de cadena a un array de caracteres
		strcpy(iris[n].label, label); //strcpy( <variable_destino>, <cadena_fuente> )
		iris[n].id = id;
		iris[n].fv[0] = v0;
		iris[n].fv[1] = v1;
		iris[n].fv[2] = v2;
		iris[n].fv[3] = v3;
		n++;
	}

	cout << "Total muestras # : " << n << endl;
	n = 0;
	for (i = 0; i <= 149; i++) {
//cout << iris[n].id iris[n].fv[0] iris[n].fv[1] iris[n].fv[2] iris[n].fv[3] << endl;
		printf("%d  %d  %d  %d  %s  %d\n", iris[i].fv[0], iris[i].fv[1],
				iris[i].fv[2], iris[i].fv[3], iris[i].label, iris[i].id);

	}
	fclose(fp);

}

//*******Ordena las instancias para cada iteracion del kfold*************
void grupos(int iteracion)
{
 int casilla[ClassNum];
 int aux;
 int numk;
 numk=SampleNum/(k*ClassNum);
 for(int j=0;j<ClassNum;j++)
	casilla[j]=j*(SampleNum/ClassNum);
 for (int clase=0;clase<ClassNum;clase++)
    {//printf("conjunto de prueba \nclase : %d \n",clase+1);
     for (int j=0;j<numk;j++)
        //iniciamos con el grupo k
        {grupok[clase][j]=casilla[clase]+iteracion*numk+j;
        //printf("%d\t",grupok[clase][j]);
        }
     //printf("\nConjunto de aprendizaje\n");
     //**************Conjunto fundamental****************
     if (iteracion<=0)
     {for (int j=numk;j<(SampleNum/ClassNum);j++)
        {
        conjfund[clase][j-numk]=casilla[clase]+j;
        //printf("%d\t",conjfund[clase][j-5]);
        }
//     printf("\n");
     }
     else
     {
     aux=0;
     for (int j=0;j<iteracion*numk;j++)
        {
        conjfund[clase][aux]=casilla[clase]+j;
        //printf("%d\t",conjfund[clase][aux]);
        aux++;
        }
     for (int j=(iteracion+1)*numk;j<(SampleNum/ClassNum);j++)
        {
        conjfund[clase][aux]=casilla[clase]+j;
        aux++;
        }
     //printf("\n");
     }
    }
}

int distan(int *ptrc,int tama){
int numk,sizcf,o,buenas=0;
int p1[FeatureVectorSize],p2[FeatureVectorSize];
float distanR[135][2];
int *ptrb;
numk=SampleNum/(k*ClassNum);
sizcf=SampleNum-(numk*ClassNum);
ptrb=(int*)conjfund;
for (int i=0;i<tama;i++)
{
    for (int m=0;m<FeatureVectorSize;m++)
        p1[m]=iris[ptrc[i]].fv[m];
    o=0;
    for (int j=0;j<sizcf;j++)
    {
        if ((j!=i)||(tama==15))
        {
            for (int m=0;m<FeatureVectorSize;m++)
                p2[m]=iris[ptrb[j]].fv[m];
            distanR[o][0]= pitagorica(p1,p2);
            if (strcmp("Setosa",iris[ptrb[j]].label)==0)
                distanR[o][1]=0;
            else if(strcmp("Versicolor",iris[ptrb[j]].label)==0)
                distanR[o][1]=1;
            else if(strcmp("Virginica",iris[ptrb[j]].label)==0)
                distanR[o][1]=2;
            o++;
        }
    }
    /*printf("%d\t",o);
    printf("%s\t",iris[ptrc[i]].label);*/
    switch(vecinos(distanR,o)){
    case 0: if (strcmp("Setosa",iris[ptrc[i]].label)==0)
                buenas++;
                break;
    case 1: if (strcmp("Versicolor",iris[ptrc[i]].label)==0)
                buenas++;
                break;
    case 2: if (strcmp("Virginica",iris[ptrc[i]].label)==0)
                buenas++;
                break;
    }
}
//printf("%d \n",buenas);
return buenas;
}

int vecinos(float orden[][2],int se)
{
    float auxd;
    int auxc,tarj,conteo[3]={0,0,0};
    for (int m=0;m<se;m++)
        {
            for (int p=m+1;p<se;p++)
            {
                if (orden[m][0]>orden[p][0])
                {
                    auxd=orden[p][0];
                    auxc=(int)orden[p][1];
                    orden[p][0]=orden[m][0];
                    orden[p][1]=orden[m][1];
                    orden[m][0]=auxd;
                    orden[m][1]=auxc;
                }
            }
        }
   /*///********
    for (int m=0;m<se;m++)
        printf("%.2f %d\t",orden[m][0],(int)orden[m][1]);
    printf("\n");
   ///*********/
    for (int m=0;m<veck;m++)
        {
            switch((int)orden[m][1]){
            case 0:conteo[0]++; break;
            case 1:conteo[1]++; break;
            case 2:conteo[2]++; break;
            }
        }
    /*for (int m=0;m<3;m++)
        printf("%d\t",conteo[m]);*/
    tarj=maximo(conteo);
   // printf("%d\n",tarj);
    return tarj;
}


float pitagorica(int x[],int y[])
{
/*float *ptry;
ptry=y;*/
float suma=0;
float raiz;
for (int u=0;u<FeatureVectorSize;u++)
{
    //printf("%.1f\t",ptry[u]);
    suma=suma+(x[u]-y[u])*(x[u]-y[u]);
}
raiz=sqrt(suma);
//printf("%.2f\n",raiz);
return raiz;
}

//revisar las condiciones de clasificacion con el k y según modificarlas
//en esta funcíón
int maximo(int revisar[3]){
 int posicion=0;
 float auxil=revisar[0];
 for (int a=2;a>=0;a--)
 {  //printf("\t%.2f",revisar[a]);
     if (auxil<revisar[a])
        {auxil=revisar[a];
         posicion=a;
        }
 }
 return posicion;
}
