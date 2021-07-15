#include <iostream>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <vector> 
#include <fstream>
using namespace std ;
int main(int argv, char *argc[])
{
    int c , k, m, i;                          //declaration of variable
    double delta_v, delta_x,dv_dt,dx_dt,h,v_next,x_next ;

    k = 20;                                      //spring constant in N/m 
    m = 20;                                      //mass in kg 
    h = 0.1;
    c = atoi(argc[1]);
    cout << "c = "<< c << endl;
    // dv_dt = (-c * v - k * x )/m;                 //detivative of w.r.t time
    // dx_dt = v ;
    double k1, k2, k3, k4;
    double w1, w2, w3, w4;

    vector<double> v;
    vector<double> x;
    vector<double> t;
    v.push_back(0);
    x.push_back(1);
    t.push_back(0);
    for(i=0;i<(15/h);i++)
    {
            k1 = h * (-c * v[i] - k * x[i] )/m;
            k2 = h * (-c * (v[i]+(k1/2)) - k * x[i] )/m;
            k3 = h * (-c * (v[i]+(k2/2)) - k * x[i] )/m;
            k4 = h * (-c * (v[i]+k3) - k * x[i])/m;
            delta_v = (k1 + 2*k2 + 2*k3 + k4)/6;
            v_next = v[i] + delta_v ;
            v.push_back(v_next);

            w1 = h * v[i] ;
            w2 = h * v[i] ;
            w3 = h * v[i] ;
            w4 = h * v[i] ;
            delta_x = (w1 + 2*w2 + 2*w3 + w4)/6;
            x_next = x[i] + delta_x ;
            x.push_back(x_next);
            t.push_back(t[i]+h);
    }

    //writing of file
    ofstream myfile;
    myfile.open("displacement.txt");
    for (i=0;i<=(15/h);i++)
    {
        myfile << x[i] << "\n";
    }
    myfile.close();

    myfile.open("time.txt");
    for (i=0;i<=(15/h);i++)
    {
        myfile << t[i] << "\n";
    }
    myfile.close();
    
    return 0;
}