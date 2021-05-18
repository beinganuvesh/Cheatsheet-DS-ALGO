#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ARRAYS
#Next Permuation in O(1) Space
def next_Permutation(arr):
    n=len(nums)
    for i in range(n-1, 0, -1):
        if arr[i-1]<arr[i]:
            break
    else:
        return arr.sort()
        
    i=i-1
    
    for j in range(n-1, 0, -1):
        if arr[i]<arr[j]:
            break
        j-=1
    
    arr[i], arr[j] = arr[j], arr[i]
    
    return arr[0:i+1]+sorted(arr[i+1:])
    
    


# In[201]:


nums = [1,3,2]
next_Permutation(nums)


# In[ ]:





# In[ ]:





# In[149]:


#ARRAYS
def inversionCount(arr, n):
    low=0
    high=n-1
    temp=[0]*n
    return mergesort(arr, low, high, temp)
 
def merge(arr, low, mid, high, temp):
    i=low
    j=mid+1
    k=low
    count=0
    
    while i<=mid and j<=high:
        if arr[i]<=arr[j]:
            temp[k]=arr[i]
            k+=1
            i+=1
        else:
            temp[k]=arr[j]
            k+=1
            j+=1
            count+=(mid-i+1)
            
    while i<=mid:
        temp[k]=arr[i]
        i+=1
        k+=1
    while j<=high:
        temp[k]=arr[j]
        j+=1
        k+=1
       
    for i in range(low, high+1):
        arr[i]=temp[i]
        
    return count
            
    
    
def mergesort(arr, low, high, temp):
    count=0
    if low<high:
        mid=low+(high-low)//2
        
        count+=mergesort(arr, low, mid, temp)
        count+=mergesort(arr, mid+1, high, temp)
        
        count+=merge(arr, low, mid, high, temp)
    
    return count


# In[150]:


n=5
a=[2, 4, 1, 3, 5]
inversionCount(a,n)


# In[ ]:





# In[ ]:





# In[1]:


#MATRIX
#Binary Search on Matrix.
def Maximum_one(row, col, mat):
    index=-1
    maximum=0
    
    for i in range(0, row):
        f=first(arr[i])
        if f!=-1:
            ones=col-f+1
            if ones>maximum:
                maximum=ones
                index=i
            
    if index!=-1:
        return index
    else:
        return -1

def first(arr):
    low=0
    high=len(arr)-1
    
    while low<=high:
        mid=low+(high-low)//2
        if (mid==0 or arr[mid-1]==0) and arr[mid]==1:
            return mid
        elif arr[mid]==0:
            low=mid+1
        else:
            high=mid     
    return -1
              


# In[ ]:





# In[ ]:





# In[153]:


#MATRIX
def Rotation(n, arr):
    for j in range(0, n):
        for i in range(n-1, -1, -1):
            print(arr[i][j], end=" ")
        print()


# In[154]:


n=3
arr=[[1, 2, 3], 
    [4, 5, 6],
    [7, 8, 9]]
Rotation(n, arr)


# In[ ]:





# In[ ]:





# In[3]:


#MATRIX AND DP
import sys
def Largest_rectangular_submatrix_with_sum_0(mat):
    n=len(mat)
    m=len(mat[0])
    temp=[0]*m
    ans=-sys.maxsize
    
    for i in range(0, n):
        temp=[0]*m
        
        for row in range(i, n):
            for col in range(0, m):
                temp[col]+=mat[row][col]
            width=Length_with_sum_0(temp)
            area=width*(row-i+1)
            if area>ans:
                ans=area
                
    return ans

def Length_with_sum_0(arr):
    ans=-sys.maxsize
    n=len(arr)
    d={0:-1}
    csum=0
    for i, val in enumerate(arr):
        csum+=val
        if csum in d:
            index=d[csum]
            ans=max(ans, i-index)
        else:
            d[csum]=i
            
    return ans


# In[120]:


a = [[9, 7, 16,  5],
    [1, -6, -7,  3],          
    [1,  8,  7,  9],          
    [7, -2,  0, 10]]
Largest_rectangular_submatrix_with_sum_0(a)


# In[ ]:





# In[ ]:





# In[98]:


#RABIN KARPS ALGO FOR PATTERN SEARCHING
def Rabin_Karp(text, pattern):
    t=len(text)
    p=len(pattern)

    hash_pattern=0
    hash_text=0
    for i in range(p):
        hash_pattern=hash_pattern+(ord(pattern[i]))
        hash_text=hash_text+(ord(text[i]))
    
    i=0
    j=p-1
    res=[]
    
    while j<t:
        if hash_text==hash_pattern:
            #Then the matching of characters start
            for x,y in zip(text[i:j+1], pattern):
                if x!=y:
                    break
            if x==text[j] and y==text[j]:
                res.append(i)
        
        i+=1
        j+=1
        if j<t:
            hash_text=hash_text-ord(text[i-1])+ord(text[j])
        else:
            break
            
    return res


# In[97]:


text='AABAACAADAABAABA'
pattern='AABA'
Rabin_Karp(text, pattern)


# In[ ]:





# In[ ]:





# In[119]:


#KMP ALGORITHM FOR PATTERN SEARCH!

#This function is to calculate the longest prefix suffix.
def lps(l, s, temp):
    i=1
    j=0
    
    while i<l:
        if s[i]==s[j]:
            temp[i]=j+1
            i+=1
            j+=1
        else:
            if j==0:
                temp[i]=0
                i+=1
            else:
                j=temp[j-1]
                
        
def KMP(text, pattern):
    t=len(text)
    p=len(pattern)
    
    temp=[0 for i in range(p)]
    #To bulid the temporary array!
    lps(p, pattern, temp)
    
    res=[]
    i=0
    j=0
    while i<t:
        if text[i]==pattern[j]:
            i+=1
            j+=1
        
        if j==p:
            res.append(i-j)
            j=temp[j-1]
        elif i<t and text[i]!=pattern[j]:
            if j==0:
                i+=1
            else:
                j=temp[j-1]
                
    return res
            


# In[120]:


text='AABAACAADAABAABA'
pattern='AABA'
KMP(text, pattern)


# In[ ]:





# In[ ]:





# In[9]:


#ARRAYS, SORTING
def Minimum_Number_of_swaps_to_sort_array(n, arr):
    temp=sorted(arr)
    d={}
    ans=0
    for i in range(0, n):
        d[arr[i]]=i
        
    for i in range(0, n):
        if arr[i]!=temp[i]:
            ans+=1
            x=arr[i]
            #Swap to the correct position
            arr[i], arr[d[temp[i]]] = arr[d[temp[i]]], arr[i]
            d[x]=d[temp[i]]
            d[temp[i]]=i

    return ans


# In[5]:


n=5
arr=[10, 19, 6, 3, 5]
Minimum_Number_of_swaps_to_sort_array(n, arr)


# In[ ]:





# In[ ]:





# In[180]:


#Binary Search
def SearchInRotated(arr, target):
    n=len(arr)
    low=0
    high=n-1
    
    while low<=high:
        mid=low+(high-low)//2

        if (arr[0]<arr[mid])==(arr[0]<target):
            if target<arr[mid]:
                high=mid
            elif target>arr[mid]:
                low=mid+1
            else:
                return mid
            
        elif arr[0]>target:
            low=mid+1
        else:
            high=mid
            
    return -1
        


# In[181]:


a=[10,11,12,13,14,15,0,1,2,3,4,5,6]
x=3
SearchInRotated(a, x)


# In[ ]:





# In[ ]:





# In[229]:


#Strings
def Balanced(s):
    l=[]
    count=0
    
    for ch in s:
        if ch=='0':
            if len(l)==0:
                l.append(ch)
            elif len(l)!=0 and l[-1]=='1':
                l.pop(-1)
                if len(l)==0:
                    count+=1
            else:
                l.append(ch)
        else:
            if len(l)==0:
                l.append(ch)
            elif len(l)!=0 and l[-1]=='0':
                l.pop(-1)
                if len(l)==0:
                    count+=1
            else:
                l.append(ch)
                
    return count
                
            


# In[199]:


s='0111100010'
Balanced(s)


# In[ ]:





# In[ ]:





# In[2]:


#Strings
import math
def countRev(s):
    l=[]
        
    for ch in s:
        if ch=='{':
            l.append(ch)
        else:
            if len(l)!=0 and l[-1]=='{':
                l.pop(-1)
            else:
                l.append(ch)
    m=0
    n=0
    for i in l:
        if i=='{':
            m+=1
        else:
            n+=1

    if len(l)%2==0:
        return math.ceil(m/2) + math.ceil(n/2)
    else:
        return -1


# In[3]:


s='}}}}}}{}{}}}{{}}}}{}}{{{}{}{}{}}{{{{}}}{}}'
countRev(s)


# In[ ]:





# In[ ]:





# In[12]:


#Strings
def SearchWordinMatrix(grid, word):
    n=len(grid)
    m=len(grid[0])
    l=len(word)
    k=0
    for i in range(n):
        for j in range(m):
            if grid[i][j]==word[0]:
                if Search(i,j,n,m,grid,word,l,k):
                    return True
    return False

def Search(i,j,n,m,grid,word,l,k):
    #Base Case
    if k==l:
        return True
    
    #Edge Cases
    if i<0 or j<0 or i>=n or j>=m:
        return False
    
    if grid[i][j]==word[k]:
        temp=grid[i][j]
        grid[i].replace(grid[i][j], '*')
        res = (Search(i+1,j,n,m,grid,word,l,k+1) or Search(i,j+1,n,m,grid,word,l,k+1) 
            or Search(i-1,j,n,m,grid,word,l,k+1) or Search(i,j-1,n,m,grid,word,l,k+1))
        grid[i].replace(grid[i][j], temp)
        return res 
    else:
        return False


# In[13]:


grid=[["axmy",
       "bgdf",
       "xeet",
       "raks"]]
word = "geeks"
SearchWordinMatrix(grid, word)


# In[ ]:





# In[ ]:





# In[7]:


#STRINGS
def RecursivlyRemove(s):
    if len(s)==0 or len(s)==1:
        return s
    
    smallerOutput=RecursivlyRemove(s[1:])
    if s[0]==smallerOutput[0]:
        return smallerOutput
    else:
        return s[0]+smallerOutput


# In[315]:


s='aabbc'
RecursivlyRemove(s)


# In[ ]:





# In[ ]:





# In[16]:


#Strings   
#ACQUIRE, SAVE AND REALESE
import sys
def findSubString(s):
    l=len(s)
    d={}
    for i in s:
        d[i]=d.get(i,0)+1
            
    ans=sys.maxsize
    m={}    
    j=0
    i=0
    while j<l:
        ch=s[j]
        m[ch]=m.get(ch,0)+1
            
        #This means you have all the characters.
        if len(m)==len(d):
            if (j-i+1)<ans:
                ans=j-i+1
                r=s[i:j+1]
                    
                #Now start releasing the characters
            while len(m)>=len(d):
                if (j-i+1)<ans:
                    ans=j-i+1
                    r=s[i:j+1]
                x=s[i]
                if x in m and m[x]==1:
                    m.pop(x)
                elif x in m and m[x]>1:
                    m[x]-=1
                i+=1
                                
        j+=1
            
    return r


# In[17]:


s='GEEKSGEEKSFOR'
findSubString(s)


# In[ ]:





# In[ ]:





# In[2]:


#Strings
#Slinding Window of dissatifaction
#Bear and Genes
from collections import Counter
import sys
def isPossible(factor, d):
    for ch in d:
        if d[ch]>factor:
            return False
    return True

def Bear_and_Genes(gene):
    n=len(gene)
    factor=n//4
    d=Counter(gene)
    ans=sys.maxsize
    
    i=j=0
    while j<n:
        if isPossible(factor, d):
            ans=min(ans, j-i)
            d[gene[i]]+=1
            i+=1
        else:
            d[gene[j]]-=1
            j+=1
            
    return ans


# In[3]:


s='GAAATAAA'
Bear_and_Genes(s)


# In[ ]:





# In[ ]:





# In[372]:


#BINARY SEARCH
def minEatingSpeed(piles, h):
    low=0
    high=sum(piles)
        
    while low<high:
        mid=low+(high-low)//2
        
        if Finish(piles, h, mid):
            high=mid
        else:
            low=mid+1
            
    return mid
      
      
def Finish(piles, h, k):
  n=len(piles)
  time=0
  
  for i in range(0, n):
    time+=math.ceil(piles[i]/k)
    if time>h:
      return False
    
  return True


# In[373]:


piles = [3,6,7,11]
h = 8
minEatingSpeed(piles, h)


# In[ ]:





# In[ ]:





# In[18]:


#BINARY SEARCH
from bisect import bisect_left, bisect_right
def Bishu(arr, m):
    n=len(arr)
    s=[0 for i in range(n+1)]
    s[1]=arr[0]
    for i in range(2, n+1):
        s[i]=arr[i-1]+s[i-1]
        
    idx=bisect_right(arr, m)
    return idx, s[idx]


# In[385]:


a=[1, 2, 3, 4, 5, 6, 7]
Bishu(a, 10)


# In[ ]:





# In[ ]:





# In[22]:


#BINARY SEARCH!
import sys
def Aggressive_Cows(arr, c):
    arr.sort()
    n=len(arr)
    low=0
    high=sum(arr)
    ans=sys.maxsize
    
    while low<=high:
        mid=low+(high-low)//2
        
        if Allowed(arr, c, mid):
            ans=mid
            low=mid+1
        else:
            high=mid-1
            
    return ans

def Allowed(arr, c, d):
    n=len(arr)
    cow_left=c-1
    left=0
    
    for i in range(1, n):
        if arr[i]-arr[left]>=d:
            left=i
            cow_left-=1
        if cow_left==0:
            return True
        
    return False


# In[23]:


a=[1,2,8,4,9]
c=3
Aggressive_Cows(a,c)


# In[ ]:





# In[ ]:





# In[31]:


#Binary Search!
def findPages(arr, n, m):
        low=0
        high=sum(arr)
        
        while low<=high:
            mid=low+(high-low)//2
            if Possible(n, arr, m , mid):
                high=mid-1
            else:
                low=mid+1
                
        return low

def Possible(n, arr, m, pages):
    students=1
    p=0
    
    for i in range(n):
        if arr[i]>pages:
            return False
        
        if arr[i]+p<=pages:
            p+=arr[i]
        else:
            students+=1
            p=arr[i]
            
        if students>m:
            return False
            
    
    return True


# In[32]:


N = 4
A = [12,34,67,90]
M = 2
findPages(A, N, M)


# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


#Binary Search
def Roti(arr, p):
    arr.sort()
    low=0
    m=max(arr)
    high=(m*(p)*(p+1))//2
    ans=0
    
    while low<=high:
        mid=low+(high-low)//2
        if CanCook(arr, p, mid):
            ans=mid
            high=mid-1
        else:
            low=mid+1       
    return ans

def CanCook(arr, p, allowed):
    n=len(arr)
    time=0
    paratha=0
    
    for i in range(0, n):
        time=arr[i]
        j=2
        while time<=allowed:
            paratha+=1
            time+=(j*arr[i])
            j+=1
        if paratha>=p:
            return True
        
    return False
    


# In[35]:


a=[1,2,3,4]
p=10
Roti(a,p)


# In[ ]:





# In[ ]:





# In[8]:


#DP
def Job_Scheduling(n, arr):
    arr.sort(key=lambda x:x[1])
    dp=[0 for i in range(n)]
    
    dp[0]=arr[0][2]
    for i in range(1, n):
        #Find the latest non-conflicting job
        profit=arr[i][2]
        for j in range(i-1, -1, -1):
            if arr[i][0]>=arr[j][1]:
                profit+=arr[j][2]
                break
                
        dp[i]=max(dp[i-1], profit)
                
    return dp[n-1]


# In[51]:


n=4
arr=[[1, 2, 50], 
    [3, 5, 20],
    [6, 19, 100],
    [2, 100, 200]]
Job_Scheduling(n, arr)


# In[ ]:





# In[ ]:





# In[37]:


#STRINGS
def chooseandswap(s):
    count=[0 for i in range(26)]
    m=0
    for ch in s:
        count[ord(ch)-ord('a')]+=1
        if count[ord(ch)-ord('a')]>m:
            m=count[ord(ch)-ord('a')]
            c=ch
    #return c          
    for i in range(0, 26):
        if count[i]>0:
            s.replace(c, chr(97+i))
            s.replace(chr(97+i), c)
            break
    return s


# In[76]:


a='baab'
b='abba'
chooseandswap(a)


# In[ ]:





# In[ ]:





# In[19]:


#DP
def count_Subsequeces_product_less_than_k(n,arr,k):
    dp=[[0 for j in range(n+1)] for i in range(k+1)]
    for i in range(1, k+1):
        for j in range(1, n+1):
            dp[i][j]=dp[i][j-1]
            if arr[j-1]<=i:
                dp[i][j]+=dp[i//arr[j-1]][j-1]+1
    return dp[k][n]    


# In[20]:


n=5
arr=[1, 2, 3, 4, 8] 
k=10
count_Subsequeces_product_less_than_k(n,arr,k)


# In[ ]:





# In[ ]:





# In[53]:


#DP
def Maximize_the_cuts(n,x,y,z):
    arr=sorted([x,y,z])
    dp=[0 for i in range(n+1)]
    
    for segment in arr:
        for i in range(1, n+1):
            if (segment==i) or (i>segment and dp[i-segment]>0):
                dp[i]=max(dp[i], 1+dp[i-segment])
    return dp[n]


# In[30]:


n = 4
x = 2
y = 1
z = 1
Maximize_the_cuts(n,x,y,z)


# In[ ]:





# In[ ]:





# In[54]:


#DP
import sys
def Minimum_Cost_to_Fill_the_Bag(n, arr, W):
    value=[]
    wt=[]
    size=0
    for i in range(0, n):
        if arr[i]!=-1:
            value.append(arr[i])
            wt.append(i+1)
            size+=1
    
    dp=[[0 for j in range(W+1)]for i in range(size+1)]
    
    for i in range(0, size+1):
        for j in range(0, W+1):
            if i==0 and j==0:
                dp[i][j]=0
            elif i==0:
                dp[i][j]=sys.maxsize
            elif j==0:
                dp[i][j]=0
            else:
                if wt[i-1]<=j:
                    dp[i][j]=min(dp[i-1][j], value[i-1]+dp[i-1][j-wt[i-1]])
                else:
                    dp[i][j]=dp[i-1][j]
                
    return dp[size][W]
    


# In[45]:


N = 5
arr = [20, 10, 4, 50, 100]
W = 5
Minimum_Cost_to_Fill_the_Bag(N, arr, W)


# In[ ]:





# In[ ]:





# In[58]:


#DP
def Count_all_Palindronic_subsequences(s,i,j):
    if i>j:
        return 0
    
    if s[i]==s[j]:
        a1=Count_all_Palindronic_subsequences(s,i+1,j)
        a2=Count_all_Palindronic_subsequences(s,i,j-1)
        return 1+a1+a2
    else:
        a1=Count_all_Palindronic_subsequences(s,i+1,j)
        a2=Count_all_Palindronic_subsequences(s,i,j-1)
        a3=Count_all_Palindronic_subsequences(s,i+1,j-1)
        return a1+a2-a3


# In[59]:


s='abcd'
Count_all_Palindronic_subsequences(s,0,len(s)-1)


# In[ ]:





# In[ ]:





# In[38]:


#DP
def optimalStrategyOfGame(arr, n):
    dp=[[0 for j in range(n)]for i in range(n)]
    for g in range(0, n):
        for i,j in zip(range(0,n) , range(g, n)):
            if g==0:
                dp[i][j]=arr[i]
            elif g==1:
                dp[i][j]=max(arr[i], arr[j])
            else:
                o1=arr[i]+min(dp[i+2][j], dp[i+1][j-1])
                o2=arr[j]+min(dp[i+1][j-1], dp[i][j-2])
                dp[i][j]=max(o1,o2)
    return dp[0][n-1]      


# In[ ]:





# In[ ]:





# In[ ]:





# In[46]:


#STACKS
#IF WE NEED CALCULATE 'TO THE RIGHT', THE MOVE FRO RIGHT TO LEFT
def NSE_RIGHT(n, arr):
    stack=[]
    res=[-1 for i in range(n)]
    res[-1]=-1
    stack.append(arr[-1])
    
    for i in range(n-2, -1, -1):
        while stack and stack[-1]>arr[i]:
            stack.pop()
            
        if stack:
            res[i]=stack[-1]
        else:
            res[i]=-1
            
        stack.append(arr[i])
        
    return res
        


# In[47]:


n=4
arr=[13, 7, 6, 12]
NSE_RIGHT(n, arr)


# In[ ]:





# In[ ]:





# In[50]:


def NSE_LEFT(n, arr):
    stack=[]
    stack.append(arr[0])
    res=[-1 for i in range(n)]
    res[0]=-1
    
    for i in range(1, n):
        while stack and stack[-1]>arr[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        else:
            res[i]=-1
        
        stack.append(arr[i])
        
    return res


# In[51]:


n=6
arr=[1, 6, 4, 10, 2, 5]
NSE_LEFT(n,arr)


# In[ ]:





# In[ ]:





# In[54]:


def NGL_RIGHT(n, arr):
    res=[-1 for i in range(n)]
    res[-1]=-1
    stack=[]
    stack.append(arr[-1])
    
    for i in range(n-2, -1, -1):
        while stack and stack[-1]<arr[i]:
            stack.pop()
        if stack:
            res[i]=stack[-1]
        else:
            res[i]=-1
        
        stack.append(arr[i])
        
    return res


# In[55]:


n=4
arr=[4, 5, 2, 25]
NGL_RIGHT(n, arr)


# In[ ]:





# In[ ]:





# In[63]:


#STACKS
def celebrity(n, arr):
    stack=[]
    for i in range(0, n):
        stack.append(i)
    while len(stack)!=1:
        p1=stack.pop(-1)
        p2=stack.pop(-1)
        
        if arr[p1][p2]==1:
            stack.append(p2)
        else:
            stack.append(p1)
            
    possible=stack[-1]
    for i in range(0,n):
        if (i!=possible) and (arr[possible][i]==1 or arr[i][possible]==0):
            return -1
        
    return possible
            

N = 3
M = [[0, 1, 0],
     [0, 0, 0], 
     [0, 1, 0]]
celebrity(N, M)


# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


#STACKS
#Maximum area of Histogram

#Apply NSE AND NSR in indexes to calculate the width of rectangle!
def MAH( arr):
    n=len(arr)
        
    left=[-1 for i in range(n)]
    stack=[]
    stack.append(0)
        
    for i in range(1, n):
        while stack and arr[stack[-1]]>=arr[i]:
            stack.pop()
                
        if stack:
            left[i]=stack[-1]
        else:
            left[i]=-1
                
        stack.append(i)
            
    
    right=[n for i in range(n)]
    stack=[]
    stack.append(n-1)
        
    for i in range(n-2, -1, -1):
        while stack and arr[stack[-1]]>=arr[i]:
            stack.pop()
            
        if stack:
            right[i]=stack[-1]
        else:
            right[i]=n
            
        stack.append(i)
            
        
    ans=-sys.maxsize
    for i in range(n):
        area=arr[i]*(right[i]-left[i]-1)
        ans=max(ans, area)
            
    return ans


# In[ ]:





# In[ ]:





# In[67]:


#STACKS AND BRACKETS
def ValidSubstring(s):
    left=0
    right=0
    for i in s:
        if s=='(':
            left+=1
        else:
            right+=1
            
        if left==right:
            ans=max(ans, 2*right)
        elif right>left:
            left=0
            right=0
            
    for i in s[::-1]:
        if s=='(':
            left+=1
        else:
            right+=1
        
        if left==right:
            ans=max(ans, 2*right)
        elif left>right:
            left=0
            right=0
            
    return ans


# In[ ]:





# In[ ]:





# In[68]:


#STACKS AND BRACKETS
def countRev (s):
    l=[]
        
    for ch in s:
        if ch=='{':
            l.append(ch)
        else:
            if len(l)!=0 and l[-1]=='{':
                l.pop(-1)
            else:
                l.append(ch)
    m=0
    n=0
    for i in l:
        if i=='{':
            m+=1
        else:
            n+=1
            
    if len(l)%2==0:
        return math.ceil(m/2)+math.ceil(n/2)
    else:
        return -1


# In[ ]:





# In[ ]:





# In[69]:


#bRACKET AND STACKS
def minimumNumberOfSwaps(s):
        r=[]
        for i in s:
            r.append(i)
        l=[]
        count=0
        
        for i in range(0, len(s)):
            if r[i]=='[':
                l.append(r[i])
            elif len(l)!=0 and l[-1]=='[':
                l.pop(-1)
            else:
                j=i+1
                while r[j]!='[':
                    j+=1
                if r[j]=='[':
                    r[i], r[j] = r[j], r[i]
                    count+=(j-i)
                    l.append(r[i])


# In[ ]:





# In[ ]:





# In[75]:


#Queues
import queue
def RottenOranges(grid):
    n=len(grid)
    m=len(grid[0])
    q=queue.Queue()
    timer=-1
    
    for i in range(n):
        for j in range(m):
            if grid[i][j]==2:
                q.put((i,j))
                grid[i][j]=3
                
    while q:
        size=q.qsize()
        for t in range(size):
            top=q.get()
            i=top[0]
            j=top[1]
            
            if isSafe(i+1, j, n, m, grid):
                grid[i+1][j]=3
                q.put((i+1,j))
                
            if isSafe(i, j+1, n, m, grid):
                grid[i][j+1]=3
                q.put((i,j+1))
                
            if isSafe(i-1, j, n, m, grid):
                grid[i-1][j]=3
                q.put((i-1,j))
                
            if isSafe(i, j-1, n, m, grid):
                grid[i][j-1]=3
                q.put((i,j-1))
                
        timer+=1
        
    return timer

def isSafe(i,j,n,m,grid):
    if i>=0 and j>=0 and i<n and j<m and grid[i][j]==1:
        return True
    else:
        return False
        


# In[ ]:


grid = [[0,1,2],[0,1,2],[2,1,1]]
RottenOranges(grid)


# In[ ]:





# In[ ]:





# In[23]:


#Queue and WHILE loop
from collections import deque
def First_Negative_interger_in_every_window_k(n, arr, k):
    q=deque()
    res=[]
    
    for i in range(0, k):
        if arr[i]<0:
            q.append(i)
        
            
    for i in range(k, n):
        #Save the answer!
        if q:
            res.append(arr[q[0]])
        else:
            res.append(0)
        
        #Remove the unwanted element from the queue.
        while q and q[0]<=i-k:
            q.popleft()
            
        #PUSH THE CURRENT ELEMENT IN THE QUEUE.
        if arr[i]<0:
            q.append(i)
    
    #For the last window
    if q:
        res.append(arr[q[0]])
    else:
        res.append(0)
            
    return res


# In[22]:


N = 5
A = [-8, 2, 3, -6, 10]
k=2
First_Negative_interger_in_every_window_k(N, A, k)


# In[ ]:





# In[ ]:





# In[27]:


#QUEUE AND WHILE LOOP
def Sum_of_minimum_and_maximum_elements_of_all_subarrays_of_size_k(arr, k):
    n=len(arr)
    s=deque() #Incresing
    g=deque() #Decreasing
    
    for i in range(0, k):
        while s and arr[s[0]]>=arr[i]:
            s.popleft()
        
        while g and arr[g[0]]<=arr[i]:
            g.popleft()
            
        s.append(i)
        g.append(i)
        
    su=0   
    for i in range(k, n):
        #Save the answer
        su+=arr[s[0]]+arr[g[0]] 
        
        #Now remove the elements which are out of window
        while s and s[0]<=i-k:
            s.popleft()
        while g and g[0]<=i-k:
            g.popleft()
            
        while s and arr[s[0]]>=arr[i]:
            s.popleft()
        while g and arr[g[0]]<=arr[i]:
            g.popleft()
            
        s.append(i)
        g.append(i)
        
    if s:
        su+=arr[s[0]]
    if g:
        su+=arr[g[0]]
        
    return su


# In[26]:


arr = [2, 5, -1, 7, -3, -1, -2]
K = 4
Sum_of_minimum_and_maximum_elements_of_all_subarrays_of_size_k(arr,K)


# In[ ]:





# In[ ]:





# In[3]:


#GREEDY
def Minimum_cost_to_cut_squares(x, y):
    m=len(x)
    n=len(y)
    # x is horizontal array
    x.sort(reverse=True)
    y.sort(reverse=True)
    
    #Now using the two pointer approach, start picking the edge which has maximum cost
    i=0
    j=0
    hori=1
    vert=1
    cost=0
    while i<m and j<n:
        if x[i]>=y[j]:
            cost+=x[i]*vert
            hori+=1
            i+=1
        else:
            cost+=y[j]*hori
            vert+=1
            j+=1
            
    total=0
    while i<m:
        total+=x[i]
        i+=1
    cost+=total*vert
    
    total=0
    while j<n:
        total+=y[j]
        j+=1
    cost+=total*hori
    return cost
            


# In[4]:


m = 6
n = 4
X = [2, 1, 3, 1, 4]
Y = [4, 1, 2]
Minimum_cost_to_cut_squares(X,Y)


# In[ ]:





# In[ ]:





# In[3]:


def Largest(n, arr):
    d={}
    i=-1
    j=-1
    l=0
    csum=0
    ans=0
    s=0
    
    while True:
        f1=False
        f2=False
        
        while j<n-1:
            f1=True
            j+=1
            s+=arr[j]
            l+=1
            d[arr[j]]=d.get(arr[j],0)+1
            
            if d[arr[j]]==2:
                break
            else:
                if s>csum:
                    csum=s
                    ans=max(ans, l)

                
        while i<j:
            f2=True
            i+=1
            s-=arr[i]
            l-=1
            d[arr[i]]=d.get(arr[i],0)-1
            
            if d[arr[i]]==1:
                break
            
            
        if f1==False and f2==False:
            break
            
            
    return ans
    


# In[4]:


n=7
a=[10, 12, 12, 10, 10, 11, 10]
Largest(n,a)

