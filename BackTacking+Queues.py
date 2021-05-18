#!/usr/bin/env python
# coding: utf-8

# In[48]:


from collections import deque
def Sum_of_Min_Max_of_K(n, arr, k):
    #Maximum
    d1=deque()
    #Minimum
    d2=deque()
    
    s=0
    for i in range(k):
        while d1 and arr[d1[-1]]>=arr[i]:
            d1.pop()
        while d2 and arr[d2[-1]]<=arr[i]:
            d2.pop()
        d1.append(i)
        d2.append(i)
            
            
    for i in range(k, n):
        s+=arr[d1[0]]+arr[d2[0]]
        
        while d1 and d1[0]<=i-k:
            d1.popleft()
        while d2 and d2[0]<=i-k:
            d2.popleft()
            
        while d1 and arr[d1[-1]]>=arr[i]:
            d1.pop()
        while d2 and arr[d2[-1]]<=arr[i]:
            d2.pop()
            
        d1.append(i)
        d2.append(i)
        
    s+=d1[0]+d2[0]
    return s
        
        
    


# In[49]:


n=7
a=[2, 5, -1, 7, -3, -1, -2]
k=4
Sum_of_Min_Max_of_K(n,a,k)


# In[ ]:





# In[ ]:





# In[42]:


def NextSmallerElement(n, arr):
    res=[-1 for i in range(n)]
    stack=[]
    stack.append(arr[-1])
    
    for i in range(n-2, -1, -1):
        
        while stack and arr[i]<stack[-1]:
            stack.pop(-1)
        
        if stack:
            res[i]=stack[-1]
        else:
            res[i]=-1
        
        stack.append(arr[i])
        
    return res


# In[43]:


n=4 
arr=[13, 7, 6, 12]
NextSmallerElement(n,arr)


# In[ ]:





# In[ ]:





# In[29]:


from collections import deque
def FirstNonRepeating(s):
    char=[0]*26
    d=deque()
    res=""
    
    for i in range(len(s)):
        d.append(s[i])
        char[ord(s[i])-ord('a')]+=1
        
        while d:
            if char[ord(d[0])-ord('a')]>1:
                d.popleft()
            else:
                break
                
        if d:
            res+=d[0]
        else:
            res+='#'
                
    return res


# In[30]:


A = "aabc"
FirstNonRepeating(A)


# In[ ]:





# In[ ]:





# In[3]:


def printFirstNegativeInteger(arr, N, K):
    r=[]
    i=0
    j=K-1
    
    while j<N:
        if j-i+1==K:
            for x in range(K):
                if arr[i+x]<0:
                    r.append(arr[i+x])
                    break
            else:
                r.append(0)
            i+=1  
            j+=1
            
    return r


# In[6]:


N = 5
A = [-8,2,3,-6,10]
K = 2
printFirstNegativeInteger(A, N, K)


# In[ ]:





# In[30]:


import queue
def ladderLength(beginWord, endWord, wordList):
        d={}
        c=0
        
        for word in wordList:
          d[word]=True
          
        q=queue.Queue()
        q.put(beginWord)
        l=len(beginWord)
        
        while q.empty() is False:
          size=q.qsize()
          for x in range(size):
                  top=q.get()

                  for i in range(l):
                    for j in range(26):
                        char=chr(ord('a')+j)
                        new_word=top[:i]+char+top[i+1:]

                        if new_word==endWord:
                            return c

                        if new_word in d:
                            q.put(new_word)
                            c+=1
                            d.pop(new_word)
                
                
        return 0


# In[31]:


beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]


# In[32]:


ladderLength(beginWord, endWord, wordList)


# In[ ]:





# In[ ]:





# In[28]:


def isSafe(n,m,i,j,grid):
    if i>=0 and i<n and j>=0 and j<m and grid[i][j]==1:
        return True
    return False       

def orangesRotting(grid):
    n=len(grid)
    m=len(grid[0])
    timer=-1
    q=queue.Queue()
    
    for i in range(n):
        for j in range(m):
            if grid[i][j]==2:
                q.put((i,j))
                grid[i][j]=3
                
    while q.empty() is False:
        size=q.qsize()
        for x in range(size):
            top=q.get()
            i=top[0]
            j=top[1]
            
            if isSafe(n,m,i-1,j,grid):
                grid[i-1][j]=3
                q.put((i-1,j))
            if isSafe(n,m,i+1,j,grid):
                grid[i+1][j]=3
                q.put((i+1,j))
            if isSafe(n,m,i,j-1,grid):
                grid[i][j-1]=3
                q.put((i,j-1))
            if isSafe(n,m,i,j+1,grid):
                grid[i][j+1]=3
                q.put((i,j+1))
                
        timer+=1
        
    for i in range(n):
        for j in range(m):
            if grid[i][j]==1:
                return -1
    return timer


# In[29]:


grid = [[0,1,2],
        [0,1,2],
        [2,1,1]]
orangesRotting(grid)


# In[ ]:





# In[ ]:





# In[51]:


def findPath(m, n):
    allpaths=[]
    path=""
    visited=[[False for j in range(n)]for i in range(n)]
    
    Helper(n, 0, 0, m, visited, allpaths, path)
    return allpaths
        
     
def isSafe(n, i, j, mat, visited, allpaths, path):
    if i<0 or j<0 or i>=n or j>=n or visited[i][j]==True or mat[i][j]==0:
        return False
    return True
    
    
def Helper(n, i, j, mat, visited, allpaths, path):
    #Edge Case
    if i<0 or j<0 or i>=n or j>=n or mat[i][j]==0 or visited[i][j]==True:
        return 
    
    #Base Case
    if i==n-1 and j==n-1:
        allpaths.append(path)
        return 
    
    visited[i][j]=True
        
    if isSafe(n, i+1, j, mat, visited, allpaths, path):
        path+='D'
        Helper(n, i+1, j, mat, visited, allpaths, path)
        path=path[:-1]
        
    if isSafe(n, i, j+1, mat, visited, allpaths, path):
        path+='R'
        Helper(n, i, j+1, mat, visited, allpaths, path)
        path=path[:-1]
        
    if isSafe(n, i-1, j, mat, visited, allpaths, path):
        path+='U'
        Helper(n, i-1, j, mat, visited, allpaths, path)
        path=path[:-1]
        
    if isSafe(n, i, j-1, mat, visited, allpaths, path):
        path+='L'
        Helper(n, i, j-1, mat, visited, allpaths, path)
        path=path[:-1]
    
    visited[i][j]=False


# In[52]:


N = 4
mat = [[1, 0, 0, 0],
        [1, 1, 0, 1], 
         [1, 1, 0, 0],
         [0, 1, 1, 1]]


# In[53]:


findPath(mat, N)


# In[ ]:





# In[ ]:





# In[77]:


def Solve(N, col, ndiag, rdiag, row, asf):
    if row==N:
        print(asf)
        return
    
    for c in range(N):
        if col[c]==False and ndiag[row+c]==False and rdiag[row-c+N-1]==False:

            col[c]=True
            ndiag[row+c]=True
            rdiag[row-c+N-1]=True
            Solve(N, col, ndiag, rdiag, row+1, asf+str(row)+"-"+str(c)+","+"")
            
            col[c]=False
            ndiag[row+c]=False
            rdiag[row-c+N-1]=False
            
    
            

def N_Queen(N):
    board=[[0 for j in range(N)]for i in range(N)]
    #Column Array
    col=[False for i in range(N)]
    #NormaL Diagonal
    ndiag=[False for i in range(2*N-1)]
    #Reverse Diagonal
    rdiag=[False for i in range(2*N-1)]
    
    return Solve(N, col, ndiag, rdiag, 0, "")
    


# In[78]:


N_Queen(4)


# In[ ]:





# In[ ]:





# In[134]:


def Helper(n, d, s, asf, l):
    if len(s)==0:
        l.append(asf)
        return
        
    for i in range(1, len(s)+1):
        left=s[:i]
        if left in d:
            right=s[i:]
            Helper(n, d, right, asf+left+" ", l)
            
            
def wordBreak(n, d, s):
        l=[]
        asf=""
        Helper(n, d, s, asf, l)
        return l
            


# In[135]:


s = "catsanddog"
n = 5 
l=[]
d = {"cats", "cat", "and", "sand", "dog"}
r=(wordBreak(n, d, s))
for i in r:
    print(i, end=" ")


# In[ ]:





# In[ ]:





# In[148]:


def Removals(s):
    l=[]
    for i in range(len(s)):
        if s[i]=='(':
            l.append(s[i])
        elif s[i]==')':
            if len(l)==0:
                l.append(s[i])
            elif l[-1]==')':
                l.append(s[i])
            elif l[-1]=='(':
                l.pop(-1)
                
    return len(l)

def Helper(s, min_removals, d, l):
    if min_removals==0:
        if s not in d and Removals(s)==0:
            l.append(s)
            d[s]=True
        return 
    
    for i in range(len(s)):
        new=s[:i]+s[i+1:]
        Helper(new, min_removals-1, d, l)
    
    

def Remove_Invalid(s):
    mr=Removals(s)
    d={}
    l=[]
    Helper(s, mr, d, l)
    return l
    


# In[149]:


s = "(a)())()"
Remove_Invalid(s)


# In[ ]:





# In[ ]:





# In[178]:


def isValid(board, i, j, n, pos):
    #Checking the entire colume.
    for r in range(0, n):
        if board[r][j]==pos:
            return False
    
    #Checking the entire row.
    for c in range(0, n):
        if board[i][c]==pos:
            return False
        
    #Checking in the sub-array.
    row=3*(i//3)
    col=3*(j//3)
    for r in range(row, row+3):
        for c in range(col, col+3):
            if board[r][c]==pos:
                return False
            
    return True 
    
def Solve(grid, row, col, n):
    #Base Case
    if row==n-1 and col==n:
        for x in range(n):
            for y in range(n):
                print(grid[x][y], end=" ")
            print()
        return 
    
    
    #If we run out of colums, then move to the next row.
    if col==n:
        col=0
        row+=1
    
    if grid[row][col]!=0:
        Solve(grid, row, col+1, n)
    else:
        for option in range(1, 10):
            if isValid(grid, row, col, n, option):
                grid[row][col]=option
                Solve(grid, row, col+1, n)
                grid[row][col]=0
    


# In[179]:


grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0],
[5, 2, 0, 0, 0, 0, 0, 0, 0],
[0, 8, 7, 0, 0, 0, 0, 3, 1],
[0, 0, 3, 0, 1, 0, 0, 8, 0],
[9, 0, 0, 8, 6, 3, 0, 0, 5],
[0, 5, 0, 0, 9, 0, 6, 0, 0],
[1, 3, 0, 0, 0, 0, 2, 5, 0],
[0, 0, 0, 0, 0, 0, 0, 7, 4],
[0, 0, 5, 2, 0, 6, 3, 0, 0]]
l=[]


# In[180]:


Solve(grid,0,0,len(grid))


# In[ ]:





# In[ ]:





# In[13]:


import heapq
def minValue(s, k):
        d={}
        for ch in s:
            d[ch]=d.get(ch,0)+1
        l=list(d.values())
        heapq._heapify_max(l)
        
        while k>0:
            top=l[0]
            heapq._heapreplace_max(l, top-1)
            k-=1
        s=0 
        for i in l:
            s+=i*i
        return s


# In[14]:


s = 'aabcbcbcabcc'
k = 3
minValue(s,k)


# In[15]:


ord('a')

