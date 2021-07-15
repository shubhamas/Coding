#include<iostream>
#include<stdio.h>
#include<string>
#include<stack>

using namespace std;

class Node
{
private:
        int operand;
        char ope;
        Node *left,*right;

public :

        Node(int operand_)
        {
                operand = operand_;
                ope = '0';
                left = NULL;
                right = NULL;
        }

        Node(char operator_)
        {
                operand = 0 ;
                ope = operator_;
                left = NULL;
                right = NULL;
        }

public :

void setleft(Node* a)
{
        left = a;
}

void setright(Node* b)
{
        right = b;
}

void setoperator(char op)
{
        ope = op;
}

void setoperand(int op_)
{
        operand = op_;
}

int getoperand()
{
        return operand;
}

Node* getright(Node* c)
{
        right = c->right;
        return right;
}

};

Node* buildinfixtree(string exp)
{
        Node *root,*r;
        stack<char> stc;
        stack<Node> stn;

        if(exp[i]=='(')
        {
                stc.push(exp[i]);
        }
        else if(isalnum(exp[i]))
        {
                int num=0;
                while(isalnum(exp[i]))
                {
                        num = (exp[i] - 48) + num *10;
                }
                r = new Node(num);
                stn.push(r);
        }
        else if(exp[i]=='+'||exp[i]=='-'||exp[i]=='*'||exp[i]=='/')
        {
                stc.push(exp[i]);
        }
        else
        {

        }

}

int main()
{
        Node *t1,*t2,*t3,*t;
        int n1,n2,n;
        char c1,c2;
        n1=10;n2=20;
        c1='a';c2='b';
        t1 = new Node(n1);
        t2 = new Node(n2);
        t3 = new Node(c1);
        t1 -> setoperator(c1);
        t2 -> setoperand(n1);
        t3 -> setleft(t1);
        t3 -> setright(t2);
        cout<< "hiii" << endl;
        t = t3->getright(t3);
        cout << t << endl;
        n = t2 -> getoperand();
        cout<< n << endl;
        return 0;

}