

def classifypoint(points,p,k=3):
    # Calculation of euclidean distance 
    for groups in points :
        for feature in points[groups]:





def main():
    # code to iniates program   
    points ={0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)], 
              1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}

    # for classification nearest neighbour 
    k = 3

    # point for testing 
    p = (2.5,7)

    print('Label of class : './format(classifypoint(points,p,k)))


if __name__ = '__main__':
    main()
