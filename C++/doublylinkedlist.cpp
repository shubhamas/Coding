#include<iostream>
using namespace std;

class Node{
        public:
        int key;
        int data;
        Node* next;
        Node* previous;

        Node()
        {
                key=0;
                data=0;
                next= NULL;
                previous=NULL;
        }

        Node(int k,int d)
        {
                key=k;
                data=d;
                next=NULL;
                previous=NULL;
        }
        
};

class Doublylinkedlist{
        public:
        Node* head;

        Doublylinkedlist(){
                head=NULL;
        }
        Doublylinkedlist(Node *n){
                head=n;
        }

        Node* nodeexists(int k)
        {
                Node* temp=NULL;
                Node* ptr=head;
                while(ptr!=NULL)
                {
                        if(ptr->key==k)
                        {
                                temp=ptr;
                        }
                        ptr=ptr->next;
                }

                return temp;
        }

        void appendNode(Node *n)
        {
                if(nodeexists(n->key))
                {
                        cout<<"Enter key alreaddy exists"<<endl;
                }
                else{
                        if(head=NULL){
                                head=n;
                                cout<<"Node appended"<<endl;
                        }
                        else{
                                Node* ptr=head;

                                while(ptr->next!=NULL){
                                        ptr=ptr->next;
                                }
                                ptr->next=n;
                                n->previous = ptr;
                                cout<<"Node appended"<<endl;
                        }
                }
        }

        void prependnode(Node *n)
        {
                if(nodeexists(n->key))
                {
                        cout<<"Node already exists"<<endl;
                }
                else{
                        head->previous=n;
                        n->next=head;

                        head=n;
                        cout<<"Node prepended"<<endl;
                }        
        }

        void insertnodeafter(int k, Node *n)
        {
                Node* ptr = nodeexists(k);

                if(ptr==NULL)
                {
                        cout<<"Insert After key:NO such key"<<endl;
                }
                else
                {
                        if(nodeexists(n->key)!=NULL)
                        cout<<"Insert Node already exists"<<endl;
                        else
                        {
                                Node* nextNode=ptr->next;
                                if(nextNode==NULL){
                                        ptr->next=n;
                                        n->previous=ptr;      
                                }else
                                {
                                        n->next= nextNode;
                                        ptr->next=n;
                                        n->previous = ptr;
                                        nextNode->previous = n;

                                        cout<<"Node inserted in between."
                                }
                                
                        }
                }
                
        }

 void deletenode(int k)
        {
                if (head==Null){
                        cout<<"List is empty"<<endl;
                }else
                {
                        if(head->key == k){
                                head = head->next;
                                head->previous=NULL;
                                cout<<"Node unlinked with keys value"<<endl;
                        }else{
                                Node* ptr = nodeexists(k);
                                Node* nextNode=ptr->next;
                                Node* prevNode = ptr->previous;
                                if (nextNode == NULL){
                                        prevNode -> next = NULL;
                                        cout<<"Node deleted at the end"<<endl;
                                }else{
                                        prevNode->next = nextNode;
                                        nextNode-> previous = prevNode;
                                        cout<<"Node deleted in Between"<<endl;
                                }
                        }
                }
                
        }

void updatenode(int k, int d)
{
        
}    

};