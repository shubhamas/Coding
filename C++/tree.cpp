#include<iostream>
#include<stdio.h>
#include<string>
#include<stack>
using namespace std;

class Node
{
    private:
    int operand_;
    char operator_;
    Node *left,*right;

    public:
    Node(int ope)
    {
        operand_ = ope;
        operator_ = '0';
        left = NULL;
        right = NULL;
    }

    Node(char op)
    {
        operand_ = 0;
        operator_ = op;
        left = NULL;
        right = NULL;
    }

    public:

    void setright(Node* right)
    {
        right = right;
    }

    void setleft(Node* left)
    {
        left = left;
    }

    int getoperand()
    {
        return operand_;
    }

    char getoperator()
    {
        return operator_;
    }

};

int buildinfixtree(string exp)
{
    int i;
    Node *t;
    stack<char> stc;
    stack<Node> stn;
    for(i=0;i<exp.length();)
    {
        if(exp[i]=='(')
        {
            stc.push(exp[i]);
        }
        else if(isalnum(exp[i]))
        {
            int num=0;
            while(isalnum(exp[i]))
            {
                num = (exp[i]-48) + num * 10;
                i = i + 1;
            }
            i = i - 1;
            t = new Node(num);
            stn.push(*t);
        }
        else if(exp[i]=='+' || exp[i]=='*' || exp[i]=='-' || exp[i]=='/')
        {
            stc.push(exp[i]);
        }
        else if(exp[i]==')')
        {
            Node *t1,*t2,*t3;
             t1 = new Node(i);
             t2 = new Node(i);
            *t1 = stn.top();
            stn.pop();
            *t2 = stn.top();
            stn.pop();
            cout<<t1->getoperand()<<endl;
            cout<<t2->getoperand()<<endl;
            t3 = new Node(stc.top());
            stc.pop();

            t3 -> setright(t1);
            t3 -> setleft(t2);

            stc.pop();

        }
        i =  i + 1;
    }
return i;
}

int main()
{
    string exp;
    int r;
    getline(cin,exp);

    r=buildinfixtree(exp);

    cout<<"r = "<<r<<endl;

    return 0;
}