#include<iostream>
#include<queue>

using namespace std;
/*class Node{
        public:
        Node *c;
        int data;
};*/
void BFS(int G[][7],int start, int n)
{
        int i=start,j;
        queue<int> q;
        int visited[7]={0};

        cout<<i<<endl;
        visited[i]=1;
        q.push(i);

        while(!q.empty())
        {
                i=q.back();
                for(j=1;j<n;j++)
                {
                        if(G[i][j]==1 && visited[j]==0)
                        {
                                cout<<j<<endl;
                                visited[j]=1;
                                q.push(j);

                        }
                }
        } 
}

int main()
{
        int G[7][7]={{0,0,0,0,0,0,0},
                     {0,0,1,1,0,0,0},
                     {0,1,0,0,1,0,0},
                     {0,1,0,0,1,0,0},
                     {0,0,1,1,0,1,1},
                     {0,0,0,0,1,0,0},
                     {0,0,0,0,1,0,0},   
                     };

        BFS(G,3,7);


        return 0;
}