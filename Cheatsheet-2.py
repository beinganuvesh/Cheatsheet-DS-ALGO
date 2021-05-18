#!/usr/bin/env python
# coding: utf-8

# In[27]:


s='anuvesh'
l=list(s)
l


# In[21]:


ar = ['aaa', 'bbb', 'ccc', 'bbb', 'aaa', 'aaa']
d={}
for ch in ar:
    d[ch]=d.get(ch,0)+1
d


# In[22]:


l=sorted(d, key=d.get, reverse=True)


# In[25]:


for i in l:
    print(i, d[i])


# In[ ]:





# In[ ]:





# In[49]:


def MinimumSwaps(n, s):
    l=[]
    cnt=0
    for i in range(0, n):
        if s[i]=='[':
            l.append(s[i])
            
        elif s[i]==']' and len(l)!=0 and l[-1]=='[':
            l.pop(-1)
            
        elif s[i]==']':
            j=i+1
            while (j<n) and (s[j]!='['):
                j+=1
            if s[j]=='[':
                s[i], s[j] = s[j], s[i]
                cnt+=1
                l.append(s[i])
                
    return cnt


# In[50]:


n=6
s='[]][]['
l=list(s)
MinimumSwaps(n, l)


# In[ ]:





# In[ ]:





# In[17]:


def Reversals(s):
    n=len(s)
    l=[]
    cnt=0
    
    if s[0]=='}':
        cnt+=1
        s[0]='{'
    if s[-1]=='{':
        cnt+=1
        s[-1]='}'
    
    for i in range(n):
        if s[i]=='{':
            l.append(s[i])
        else:
            if len(l)!=0 and l[-1]=='{':
                l.pop(-1)
            else:
                l.append(s[i])
    
    if len(l)%2==0:
        return cnt+(len(l)//2)
    else:
        return -1


# In[18]:


s='}{{}}{{{'
l=list(s)
Reversals(l)


# In[ ]:





# In[ ]:





# In[10]:


import sys
def AggressiveCows(n, arr, c):
    arr.sort()
    ans=-sys.maxsize
    low=arr[0]
    high=arr[-1]
    
    
    while low<=high:
        gap=low+(high-low)//2
        cow_left=c-1
        
        left=0
        for i in range(1, n):
            if arr[i]-arr[left]>=gap:
                left=i
                cow_left-=1
            if cow_left==0:
                ans=max(ans, gap)
                break
        
        if cow_left>0:
            high=gap-1
        else:
            low=gap+1
            
    return ans
            
        


# In[11]:


n=4
a=[2,4,8,16]
c=3
AggressiveCows(n, a, c)


# In[ ]:





# In[ ]:





# In[86]:


def KOKO_EATING_BANANAS(arr, H):
        n=len(arr)
        arr.sort()
        
        high=arr[-1]
        low=1
        
        while low<=high:
            k=low+(high-low)//2
            
            #Count the time taken to eat bananas with speed K.
            time_taken=0
            for i in range(n):
                time_taken+= arr[i]//k
                if arr[i]%k!=0:
                    time_taken+=1
                
            if time_taken<=H:
                high=k-1
            elif time_taken>H:
                low=k+1
                
        return low
                
                


# In[87]:


p=[312884470]
H=312884469
minEatingSpeed(p, H)


# In[ ]:





# In[ ]:





# In[15]:


def FractionalKnapsack(n, W, val, weight):
    res=[]
    for i in range(n):
        x=val[i]/weight[i]
        res.append((val[i], weight[i], x))
        
    r=sorted(res, key=lambda x:x[2], reverse=True)
    
    val=0
    i=0
    while i<n and W>=0:
        if r[i][1]<=W:
            val+=r[i][0]
            W-=r[i][1]
        else:
            val+=(W*r[i][2])
            W=0
        i+=1
    return val
        


# In[14]:


N = 3
W = 50
values = [60,100,120]
weight = [10,20,30]
FractionalKnapsack(N, W, values, weight)


# In[ ]:





# In[ ]:





# In[16]:


def Maximum_Trains(n, m, arr):
    #Sorting the array on basis of departure time/ ending time!
    arr=sorted(arr, key=lambda x:x[1])
    
    #Choosing the first train!
    ending_time=arr[0][1]
    last_platform=arr[0][2]
    cnt=1
    
    #Using dictionaries to store the departure time of last train of specific platform!
    #Using key as platform and departure time as value!
    d={}
    d[last_platform]=ending_time
    
    
    for i in range(1, m):
        if arr[i][2] in d:
            if arr[i][0]>=d[arr[i][2]]:
                cnt+=1
                d[arr[i][2]]=arr[i][1]
        else:
            cnt+=1
            d[arr[i][2]]=arr[i][1]
            
    return cnt
        


# In[17]:


n=2
m=5
a= [[1000, 1030, 1],
    [1010, 1020, 1], 
    [1025, 1040, 1], 
    [1130, 1145, 2], 
    [1130, 1140, 2 ]]
Maximum_Trains(n,m,a)


# In[18]:


n=3
m=6
arr = [[1000, 1030, 1],
       [1010, 1030, 1],
       [1000, 1020, 2],
       [1030, 1230, 2],
       [1200, 1230, 3],
       [900, 1005,1]]
Maximum_Trains(n, m, arr)


# In[ ]:





# In[ ]:





# In[22]:


def minimumPlatform(n,arr,dep):
    res=[]
    for i in range(n):
        res.append((arr[i], dep[i]))
    r=sorted(res, key=lambda x:x[1])
    
    platform=1
    
    d={platform : r[0][1]}
    for i in range(1, n):
        
        for key in d:
            if r[i][0]>=d[key]:
                d[key]=r[i][1]
                break
        else:
            platform+=1
            d[platform]=r[i][1]
        
    return len(d)


# In[23]:


N = 6 
arr=[900,  940, 950,  1100, 1500, 1800]
dep=[910, 1200, 1120, 1130, 1900, 2000]
minimumPlatform(N, arr, dep)


# In[ ]:





# In[ ]:





# In[52]:


def fourSum(arr, m):
    n=len(arr)
    arr.sort()
    res=[]
    
    for i in range(0, n-4):
        for j in range(i+1, n-3):
            left=j+1
            right=n-1
            while left<right:
                s=arr[i]+arr[j]+arr[left]+arr[right]  
                if s==m:
                    res.append((arr[i], arr[j], arr[left], arr[right]))
                    left+=1
                    right-=1
                elif s>m:
                    right-=1
                else:
                    left+=1
                    
    return res
    


# In[53]:


N = 7
K = 23
A = [10,2,3,4,5,7,8]`
fourSum(A, K)


# In[ ]:





# In[ ]:





# In[60]:


def Theif(n, arr, i, dp):
    if i>=n:
        return 0
    
    if dp[i+1]==-1:
        x=Theif(n, arr, i+1, dp)
    else:
        x=dp[i+1]
        
    if dp[i+2]==-1:
        y=arr[i] + Theif(n, arr, i+2, dp)
    else:
        y=dp[i+2]
    return max(x,y)


# In[ ]:


def Theif_DP(n, arr):
    dp=[-1 for i in range(n+1)]
    


# In[61]:


n = 6
a = [5,5,10,100,10,5]
dp=[-1 for i in range(n+2)]
Theif(n, a, 0, dp)


# In[ ]:





# In[ ]:





# In[103]:


def Solve(M, n, arr):
    w=0
    l=[]
    d={}
    
    for i in arr:
        d[i]=True
    
    for i in range(1, M+1):
        if i not in d:
            if (w+i)<=M:
                l.append(i)
                w+=i
    return (l)


# In[104]:


M=10
n=4
arr=[1,2,3,4]
Solve(M, n, arr)


# In[ ]:





# In[ ]:





# In[2]:


def StocksPurchase(n, arr, k):
    cnt=0
    for i in range(n):
        allowed=i+1
        if k>=0:
            if (k//arr[i])<=allowed:
                cnt+=k//arr[i]
                k-=arr[i]*(k//arr[i])
            else:
                cnt+=allowed
                k-=arr[i]*allowed
            
    return cnt


# In[12]:


def Stock_Greedy(n, a, k):
    res=[]
    cnt=0
    for i in range(n):
        res.append((a[i], i+1))
    arr=sorted(res, key=lambda x:x[0])
    
    i=0
    while i<n and k>=0:
        if (k//arr[i][0])<=arr[i][1]:
            cnt+=k//arr[i][0]
            k-=(k//arr[i][0])*arr[i][0]
        else:
            cnt+=arr[i][1]
            k-=arr[i][1]*arr[i][0]
        i+=1
            
    return cnt
        


# In[14]:


n=3
price = [7, 10, 4]
k = 100
Stock_Greedy(n, price, k)


# In[ ]:





# In[ ]:





# In[23]:


def Candies(n, a, k):
    arr=sorted(a)
    c1, c2 = 0, 0
    i=0
    j=n-1
    
    while i<=j:
        c1+=arr[i]
        i+=1
        j-=k
        
    i=0
    j=n-1
    while i<=j:
        c2+=arr[j]
        j-=1
        i+=k
    
    return c1,c2


# In[24]:


n=4
k=2
a=[3,2,1,4]
Candies(n, a, k)


# In[ ]:





# In[ ]:





# In[28]:


def Solve(n, arr):
    arr.sort()
    li=[]
    for j in range(n//2):
        li.append(arr[j])
        li.append(arr[n-1-j])
        
    if(n%2!=0):
        li.append(arr[n//2])
        
    sum=0
    for i in range(n-1):
        x=abs(li[i]-li[i+1])
        sum=sum+x
    y=abs(li[0]-li[n-1])
        
    print(sum+y)


# In[30]:


n=4
a=[4,2,1,8]
Solve(n, a)


# In[ ]:





# In[ ]:





# In[ ]:


def PrintRecursively(l, i, j):
    
    #Base Case
    if i==m-1:
        print(output)
        return 
        
    #Call for other Words.
    for y in range(n):
        PrintRecursively(l, i+1, j)
        
    
def printWords(arr):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            PrintRecursively(arr, i, j)
    
        


# In[32]:


l=[["you", "we"],
   ["have", "are"],
   ["sleep", "eat", "drink"]]


# In[ ]:





# In[ ]:





# In[12]:


def Minimum_Swaps(n, arr):
    t=[]
    for i in range(n):
        t.append([arr[i], i])
    r=sorted(t, key=lambda x:x[0])
    cnt=-1
    
    i=0
    while i<n:
        #check if the element is situated at its correct position.
        if i==r[i][1]:
            i+=1
            continue
        else:
            #Swap the elements an bring them at correct position.
            r[i][0], r[r[i][1]][0] = r[r[i][1]][0], r[i][0]
            r[i][1], r[r[i][1]][1] = r[r[i][1]][1], r[i][1]
        
        if i==r[i][1]:
            i-=1
        
        i+=1
        cnt+=1
        
        
    return cnt
        


# In[13]:


n=4
arr=[1,4,3,2]
Minimum_Swaps(n, arr)


# In[ ]:





# In[ ]:





# In[25]:


import sys
def findSubString(s):
    r=len(set(s))
    n=len(s)
    d={}
    ans=sys.maxsize
    
    i=0
    j=0
    while j<n:
        ch=s[j]
        d[ch]=d.get(ch,0)+1
        
        if len(d)==r:
            le=(j-i+1)
            ans=min(ans, le)
            
            #Now start releasing the characters!
            while i<j:
                new_char=s[i]
                if d[new_char]==1:
                    le=j-i+1
                    ans=min(ans, le)
                    d.pop(new_char)
                    i+=1
                    break
                else:
                    d[new_char]-=1
                i+=1
                
        j+=1
        
    return ans
    


# In[28]:


s='GEEKSGEEKSFOR'
findSubString(s)


# In[ ]:





# In[ ]:





# In[56]:


def nextPermutation(n, arr):
    j=-1
    i=n-1
    #Search for the element which breaks the trend
    while i>0:
        if arr[i-1]<arr[i]:
            j=i-1
            break
        i-=1
            
    #Now swap the position of the element
    for i in range(n-1, -1, -1):
        if arr[i]>arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
            #Now sort the remaining part
            arr[j+1:] = sorted(arr[j+1:])
            return arr


# In[57]:


n = 6
arr = [1, 2, 3, 6, 5, 4]
nextPermutation(n, arr)


# In[ ]:





# In[ ]:





# In[61]:


def SmallestSubset(n, arr):
    arr=sorted(arr, reverse=True)
    total=sum(arr)
    current=0
    cnt=0
    for i in range(n):
        current+=arr[i]
        cnt+=1
        rest=total-current
        
        if current>=rest:
            return cnt


# In[63]:


n=4
arr = [3, 1, 7, 1]
SmallestSubset(n, arr)


# In[ ]:





# In[ ]:





# In[36]:


import math
def CountTheReversals(s):
    n=len(s)
    r=""
    l=[]
        
    for i in s:
        if i=='{':
            l.append(i)
        else:
            if len(l)!=0 and l[-1]=='{':
                l.pop(-1)
            else:
                r+=i
    for i in l:
        r+=i
    m=0
    n=0
    for i in r:
        if i=='{':
            m+=1
        else:
            n+=1
    if len(r)%2==0:
        return math.ceil(m/2)+math.ceil(n/2)
    else:
        return -1


# In[37]:


s='{{}{{{}{{}}{{'
CountTheReversals((s))


# In[ ]:





# In[ ]:





# In[45]:


def SentenceintoMobileKEYPAD(s):
    n=len(s)
    d={'A': '2', 'B': '22', 'C':'222', 'D':'3', 'E':'33', 'F':'333', 'G':'4', 'H':'44', 'I':'444',
       'J':'5', 'K':'55', 'L':'555', 'M':'6', 'N':'66', 'O':'666', 'P':'7', 'Q':'77', 'R':'777', 'S':'7777',
      'T':'8', 'U':'88', 'V':'888', 'W':'9', 'X':'99', 'Y':'999', 'Z':'9999', ' ' : '0'}
    
    r=""
    for i in s:
        r+=d[i]
        
    return int(r)


# In[46]:


s='HELLO WORLD'
SentenceintoMobileKEYPAD(s)


# In[ ]:





# In[ ]:





# In[101]:


def Check(n, m, grid, word, wordlen, i, j, k):
    flag=False
    
    if i>=0 and j>=0 and i<n and j<m and grid[i][j]==word[k]:
        temp=grid[i][j]
        grid[i][j]=0
        k+=1
        
        if wordlen==k:
            flag=True
        else:
            a=Check(n, m, grid, word, wordlen, i+1, j, k)
            b=Check(n, m, grid, word, wordlen, i, j+1, k)
            c=Check(n, m, grid, word, wordlen, i-1, j, k)
            d=Check(n, m, grid, word, wordlen, i, j-1, k)
            e=Check(n, m, grid, word, wordlen, i+1, j+1, k)
            f=Check(n, m, grid, word, wordlen, i-1, j-1, k)
            g=Check(n, m, grid, word, wordlen, i-1, j+1, k)
            h=Check(n, m, grid, word, wordlen, i+1, j-1, k)
            flag=(a or b or c or d or e or f or g or h)
            
        grid[i][j]=temp
        
    return flag
            
        
def Solve(grid, word):
    wordlen=len(word)
    n=len(grid) 
    m=len(grid[0]) 
    l=[]
    for i in range(n):
        for j in range(m):
            ispresent=Check(n, m, grid, word, wordlen, i, j, 0)
            if ispresent:  
                l.append((i,j))

    return l       


# In[102]:


G=[list("GEEKSFORGEEKS"),
      list("GEEKSQUIZGEEK"),
       list("IDEQAPRACTICE")]
x = "GEEKS"
G


# In[103]:


Solve(G, x)


# In[ ]:





# In[ ]:





# In[105]:


def Checking_word(i, j, n, m, grid, word, wordlen, k):
    found=0
    if i>=0 and j>=0 and i<n and j<m and grid[i][j]==word[k]:
        temp=grid[i][j]
        grid[i][j]=0
        k+=1
        if k==wordlen:
            found=1
        else:
            found+=Checking_word(i+1, j, n, m, grid, word, wordlen, k)
            found+=Checking_word(i, j+1, n, m, grid, word, wordlen, k)
            found+=Checking_word(i-1, j, n, m, grid, word, wordlen, k)
            found+=Checking_word(i, j-1, n, m, grid, word, wordlen, k)
        
        grid[i][j]=temp
    
    return found
        
def Count_of_number_of_given_string(row, col, grid, word):
    wordlen=len(word)
    cnt=0
    
    for i in range(row):
        for j in range(col):
            cnt+=Checking_word(i, j, row, col, grid, word, wordlen, 0)
                
    return cnt


# In[107]:


a= [['D','D','D','G','D','D'],
    ['B','B','D','E','B','S'],
    ['B','S','K','E','B','K'],
    ['D','D','D','D','D','E'],
    ['D','D','D','D','D','E'],
    ['D','D','D','D','D','G']
]
s= "GEEKS"
Count_of_number_of_given_string(len(a), len(a[0]), a, s)


# In[ ]:





# In[ ]:





# In[9]:


def RemoveConsecutive(s):
    if len(s)==1:
        return s
    
    smallerOutput=RemoveConsecutive(s[1:])
    
    if s[0]==smallerOutput[0]:
        return smallerOutput
    else:
        return s[0]+smallerOutput
    


# In[11]:


s='aabaa'
RemoveConsecutive(s)


# In[ ]:





# In[ ]:





# In[30]:


def WildcardMatching(string, pattern):
    n=len(string)
    m=len(pattern)
    dp=[[False for j in range(m+1)]for i in range(n+1)]
    
    for i in range(n+1):
        for j in range(m+1):
            
            if i==0 and j==0:
                dp[i][j]=True
                
            elif i==0:
                dp[i][j]=False
                
            elif j==0:
                if pattern[j]=='*':
                    dp[i][j]=dp[i-1][j]
                else:
                    dp[i][j]=False
                
            else:
                if string[i-1]==pattern[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                    
                elif pattern[j-1]=='?':
                    dp[i][j]=dp[i-1][j-1]
                    
                elif pattern[j-1]=='*':
                    dp[i][j]=dp[i-1][j] | dp[i][j-1]
                    
                else:
                    dp[i][j]=False
                    
    return dp[n][m]
                


# In[31]:


pattern='ge?ks*'
string='geeksforgeeks'
WildcardMatching(string,pattern)


# In[ ]:





# In[ ]:





# In[35]:


import sys
def smallestWindow(s, p):
    d={}
    m={}
    req={}
    l=sys.maxsize
    ans=""

    n=len(s)
    
    for i in p:
        m[i]=m.get(i,0)+1
        
    for i in p:
        req[i]=req.get(i,0)+1
        
    i=0
    j=0
    while j<n:
        ch=s[j]
        d[ch]=d.get(ch,0)+1
        
        if ch in m:
            if m[ch]==1:
                m.pop(ch)
            else:
                m[ch]-=1
                
        if len(m)==0:
            #Save the answer
            if (j-i+1)<l:
                ans=s[i:j+1]
                l=j-i+1
            
            #Now Release!
            while i<j:
                x=s[i]
                if x not in req:
                    if x in d:
                        if d[x]==1:
                            d.pop(x)
                        else:
                            d[x]-=1
                else:
                    if d[x]>req[x]:
                        if d[x]==1:
                            d.pop(x)
                        else:
                            d[x]-=1
                    else:
                        break
                i+=1
            
            if j-i+1<l:
                ans=s[i:j+1]
                l=j-i+1
        
        j+=1  
        
    return ans


# In[36]:


S = 'DBAECBBABDCAAFBDDCABGBA'
P = 'ABBCDC'
smallestWindow(S,P)


# In[ ]:





# In[ ]:





# In[4]:


def Celebrity(n, mat):
    stack=[]
    #Push all the people into the stack.
    for i in range(n):
        stack.append(i)
        
    while len(stack)!=1:
        p1=stack.pop()
        p2=stack.pop()
        
        if mat[p1][p2]==1:
            #If p1 knows p2, then p1 cant be a celebrity.
            stack.append(p2)
        elif mat[p1][p2]==0:
            #If p1 dont know p2, then p2 cant be a celebrity.
            stack.append(p1)
            
    celeb=stack[-1]
    for col in range(n):
        if mat[celeb][col]==1:
            return -1
    
    return celeb
        


# In[5]:


N = 3
M = [[0, 1, 0],
     [0, 0, 0], 
     [0, 1, 0]]
Celebrity(N, M)


# In[ ]:





# In[ ]:





# In[2]:


def Position(n, arr, pos, x):
    after=arr[pos-1:]
    before=arr[0:pos-1]+[x]
    return before+after


# In[3]:


n=10
arr=[1,2,3,4,5,6,7,8,9,10]
pos=9
x=1
Position(n, arr, pos, x)


# In[ ]:





# In[ ]:





# In[23]:


def EvaluatePostfix(s):
    res=0
    l=[]
    flag=False
    
    for i in s:
        if i=='/':
            if flag==False:
                e1=l.pop()
                e2=l.pop()
                res+=e1/e2
                flag=True
            else:
                e=l.pop()
                res/=e
                
        elif i=='*':
            if flag==False:
                e1=l.pop()
                e2=l.pop()
                res+=e1*e2
                flag=True
            else:
                e=l.pop()
                res*=e
            
        elif i=='+':
            if flag==False:
                e1=l.pop()
                e2=l.pop()
                res=e1+e2
                flag=True
            else:
                e=l.pop()
                res+=e
            
        elif i=='-':
            if flag==False:
                e1=l.pop()
                e2=l.pop()
                res=e1-e2
                flag=True
            else:
                e=l.pop()
                res-=e
                
        else:
            l.append(int(i))
            
    return res


# In[24]:


S="231*+9-"
EvaluatePostfix(S)


# In[ ]:





# In[ ]:





# In[54]:


import sys
def MAH(n, arr):
    #Find nearest smaller to left and nearest smaller to right
    ans=-sys.maxsize
    
    right=[n for i in range(n)]
    stack=[]
    stack.append(n-1)
    for i in range(n-2, -1, -1):
        while (len(stack)>0 and arr[i]<stack[-1]):
            stack.pop()
            
        if len(stack)==0:
            right[i]=n
        else:
            right[i]=stack[-1]
        stack.append(arr[i])
        
        
    left=[-1 for i in range(n)]
    stack=[]
    stack.append(0)
    for i in range(1, n):
        while (len(stack)>0 and arr[i]<stack[-1]):
            stack.pop()
            
        if len(stack)==0:
            left[i]=-1
        else:
            left[i]=stack[-1] 
        stack.append(arr[i])
        
    for i in range(n):
        nsl=left[i]
        nrl=right[i]
        x=(nrl-nsl-1)
        area=x*arr[i]
        ans=max(ans, area)
    
    return ans
        
        


# In[55]:


N = 6
arr = [6,2,5,4,5,1,6]
MAH(N, arr)


# In[ ]:





# In[ ]:





# In[58]:


def checkRedundantBrackets(expression) :
    stack=[]
    for i in expression:
        if i!=')':
            stack.append(i)
        else:
            count=0
            while len(stack)!=0 and stack.pop()!='(':
                count+=1
            if count==0:
                return True
            
    return False


# In[ ]:





# In[ ]:





# In[36]:


class CircularQueue:
    def __init__(self, size):
        self.size=size
        self.queue=[-1 for i in range(size)]
        self.front=0
        self.rear=0
        
    def Front(self):
        return self.queue[self.front]
        
    def Rear(self):
        return self.queue[self.rear-1]
        
    def enQueue(self, value):
        #Check if the queue us empty
        if (self.rear==self.size-1 and self.front==0):
            print('Queue is Full')
            return 
        
        if (self.rear+1)>=self.size:
            self.rear=(self.rear)%(self.size-1)
        if (self.front+1)>=self.size:
            self.front=(self.front)%(self.size-1)
            
        self.queue[self.rear]=value
        self.rear+=1
            
    def deQueue(self):
        #Check if queue is empty or not.
        if self.front==self.rear:
            print('Queue is empty')
            return 
        
        self.queue[self.front]=-1
        self.front+=1
        
        


# In[ ]:





# In[ ]:





# In[14]:


def knapSack(W, wt, val, n):
    dp=[[0 for j in range(W+1)]for i in range(n+1)]
    
    for i in range(1, n+1):
        for j in range(1, W+1):
            if wt[i-1]<=j:
                dp[i][j]=max(val[i-1]+dp[i-1][j-wt[i-1]], dp[i-1][j])
            else:
                dp[i][j]=dp[i-1][j]
                
    return dp[n][W]


# In[15]:


N = 3
maxW = 4
values = [1,2,3]
weight = [4,5,1]
knapSack(maxW, weight, values, N)


# In[ ]:





# In[ ]:





# In[28]:


def nCr(n, r):
    mod=(10**9)+7
    dp=[[0 for j in range(r+1)]for i in range(n+1)]
        
    #When r=0.
    for i in range(0, n+1):
        dp[i][0]=1
            
    for i in range(1, n+1):
        for j in range(1, r+1):
            if i>=j:
                dp[i][j]=dp[i-1][j]+dp[i-1][j-1]
            else:
                dp[i][j]=0
                    
    return dp[n][r]


# In[27]:


n=6
r=4
nCr(n, r)


# In[ ]:





# In[ ]:





# In[30]:


def nPr(n, r):
    dp=[[0 for j in range(r+1)]for i in range(n+1)]
    
    for i in range(0, n+1):
        dp[i][0]=1
        
    for i in range(1, n+1):
        for j in range(1, r+1):
            if i>=j:
                dp[i][j]=dp[i-1][j]+(j)*dp[i-1][j-1]
            else:
                dp[i][j]=0
                
    return dp[n][r]
            


# In[31]:


nPr(10, 3)


# In[ ]:





# In[ ]:





# In[33]:


def MCM(n, arr):
    dp=[[0 for j in range(n-1)]for i in range(n-1)]
    
    for g in range(0, len(dp)):
        for i, j in zip(range(0, len(dp)), range(g, len(dp))):
            if g==0:
                dp[i][j]=0
            elif g==1:
                dp[i][j]=arr[i]*arr[j]*arr[j+1]
            else:
                ans=sys.maxsize
                for k in range(i, j):
                    left=dp[i][k]
                    right=dp[k+1][j]
                    main=arr[i]*arr[j+1]*arr[k+1]
                    total=left+right+main
                    ans=min(ans, total)
                    dp[i][j]=ans
                    
    return dp[0][len(dp)-1]
                


# In[ ]:





# In[ ]:





# In[55]:


def findLongestChain(pairs):
    n=len(pairs)
    arr=sorted(pairs, key=lambda x:x[1])
    dp=[1 for i in range(n)]
    
    for i in range(1, n):
        for j in range(0, i):
            if arr[i][0]>arr[j][1]:
                if dp[j]>=dp[i]:
                    dp[i]=dp[j]+1
                      
    return dp


# In[56]:


p = [[1,2], [2,3], [3,4]]
findLongestChain(p)


# In[ ]:





# In[ ]:





# In[17]:


def editDistance(s, t):
	n=len(s)
	m=len(t)
	dp=[[0 for j in range(m+1)]for i in range(n+1)]
	
	
	for i in range(n+1):
	    for j in range(m+1):
	        if i==0 and j==0:
	            dp[i][j]=0
	        elif i==0:
	            dp[i][j]=j
	        elif j==0:
	            dp[i][j]=i
	        else:
	            if s[i-1]==t[j-1]:
	                dp[i][j]=dp[i-1][j-1]
	            else:
	                dp[i][j]=1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
	                
	return dp


# In[18]:


s = "geek"
t = "gesek"
editDistance(s, t)


# In[ ]:





# In[ ]:





# In[42]:


def countFriendsPairings(n):
    dp=[0 for i in range(n+1)]
    for i in range(n+1):
        #BASE CASE.
        if i<=2:
            dp[i]=i
        else:
            dp[i] = dp[i-1] + (i-1)*dp[i-2]
            
    return dp[n]


# In[25]:


countFriendsPairings(4)


# In[ ]:





# In[ ]:





# In[42]:


def GoldMine(n, m, arr):
    dp=[[0 for j in range(m)]for i in range(n)]
    for j in range(0, m):
        for i in range(0, n):
            if j==0:
                dp[i][j]=arr[i][j]
            elif i==0:
                dp[i][j]=max(dp[i][j-1] , dp[i+1][j-1]) + arr[i][j]
            elif i==n-1:
                dp[i][j]=max(dp[i][j-1], dp[i-1][j-1]) + arr[i][j]
            else:
                dp[i][j]=max(dp[i-1][j-1], dp[i][j-1], dp[i+1][j-1]) + arr[i][j]
                
    return dp[0][m-1]
                


# In[43]:


n=4
m=4
mat = [[10, 33, 13, 15],
        [22, 21, 4, 1],
        [5, 0, 2, 3],
        [0, 6, 14, 2]]
GoldMine(n, m, mat)


# In[ ]:





# In[ ]:





# In[1]:


def countWays(n, k):
    same=[0 for i in range(n+1)]
    diff=[0 for i in range(n+1)]
    total=[0 for i in range(n+1)]
     
    #Base Case!
    same[2]=k*1
    diff[2]=k*(k-1)
    total[2]=same[2]+diff[2]
        
    for i in range(3, n+1):
        same[i]=diff[i-1]
        diff[i]=total[i-1]*(k-1)
        total[i]=same[i]+diff[i]
            
    return total[n]


# In[2]:


N=17
K=1
countWays(N, K)


# In[ ]:





# In[ ]:





# In[33]:


import sys
def CutTheSegments(n, x, y, z):
    #Base case
    if n<=0:
        return 0
    
    ans=-sys.maxsize
    a1=a2=a3=0
    
    for i in range(0, n+1):
        if i==x:
            a1=1+CutTheSegments(n-x, x, y, z)
        elif i==y:
            a2=1+CutTheSegments(n-y, x, y, z)
        elif i==z:
            a3=1+CutTheSegments(n-z, x, y, z)
            
        ans=max(ans, a1, a2, a3)
        
    return ans
    


# In[35]:


N = 5
x = 5 
y = 3
z = 2
CutTheSegments(N, x, y, z)


# In[40]:


def CutTheSegments_DP(n, x, y, z):
    dp=[0 for i in range(n+1)]
    arr=sorted([x,y,z])
    
    for segment in arr:
        for i in range(1, n+1):
            if i==segment or (i>segment and dp[i-segment])>0:
                dp[i]=max(dp[i], 1+dp[i-segment])
                
    return dp[n]  


# In[41]:


N = 5
x = 5 
y = 3
z = 2
CutTheSegments_DP(N, x, y, z)


# In[ ]:





# In[ ]:





# In[50]:


def LCS_Of_three(l, m, n, x, y, z):
    dp=[[[0 for k in range(n+1)]for j in range(m+1)]for i in range(l+1)]
    for i in range(1, l+1):
        for j in range(1, m+1):
            for k in range(n+1):
                if x[i-1]==y[j-1]==z[k-1]:
                    dp[i][j][k]=1+dp[i-1][j-1][k-1]
                else:
                    dp[i][j][k]=max(dp[i-1][j][k], dp[i][j-1][k], dp[i][j][k-1])
                    
    return dp[l][m][n]


# In[51]:


l=5 
m=8 
n=13
x='geeks' 
y='geeksfor' 
z='geeksforgeeks'
LCS_Of_three(l, m, n, x, y, z)


# In[ ]:





# In[ ]:





# In[57]:


def Count_of_Subsequence_product_less_than_k(n, arr, k):
    dp=[[0 for j in range(n+1)]for i in range(k+1)]
    
    for i in range(1, k+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i][j-1]
            
            if arr[j-1]<=i:
                dp[i][j]+=dp[i//arr[j-1]][j-1]+1
                
    return dp[k][n]
                


# In[58]:


n=4
arr=[1, 2, 3, 4] 
k = 10
Count_of_Subsequence_product_less_than_k(n, arr, k)


# In[ ]:





# In[ ]:





# In[69]:


def Longest_subsequence(n, arr):
    dp=[1 for i in range(n)]

    for i in range(1, n):
        for j in range(0, i):
            if (arr[i]>arr[j] and arr[i]-arr[j]==1) or (arr[j]>arr[i] and arr[j]-arr[i]==1):
                if dp[j]+1>dp[i]:
                    dp[i]=dp[j]+1
    return max(dp)
                    


# In[70]:


N = 5
A = [1, 2, 3, 4, 5]
Longest_subsequence(N, A)


# In[ ]:





# In[ ]:





# In[80]:


def Max_Sum_no_three_consecutives(n, arr):
    dp=[0 for i in range(n)]
    
    #Base Cases!
    dp[0]=arr[0]
    dp[1]=arr[0]+arr[1]
    dp[2]=max(dp[1], arr[2]+arr[0], arr[2]+arr[1])
    
    for i in range(3, n):
        exclude_current=dp[i-1]
        exclude_last=arr[i]+dp[i-2]
        exclude_second_last=arr[i]+arr[i-1]+dp[i-3]
        
        dp[i]=max(exclude_current, exclude_last, exclude_second_last)
    
    return dp[n-1]


# In[81]:


n=5
a=[3000, 2000, 1000, 3, 10]
Max_Sum_no_three_consecutives(n, a)


# In[ ]:





# In[ ]:





# In[93]:


def maxSquare(n, m, mat):
    dp=[[0 for j in range(m+1)]for i in range(n+1)] 
    for i in range(1, n+1):
        for j in range(1, m+1):
            if i==j and mat[i-1][j-1]==1:
                dp[i][j]=1+dp[i-1][j-1]
            else:
                dp[i][j]=max(dp[i-1][j], dp[i][j-1])
                    
    return dp


# In[94]:


n=3
m=3
a=[[1, 1, 1],
   [1, 1, 1],
    [1, 1, 1]]
print(maxSquare(n, m, a))

