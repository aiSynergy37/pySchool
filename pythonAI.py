# finding the sqrt

# binarySearch--iterative way ---sorted array

# LeetCode1


class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        left, right = 1, x
        while left <= right:
            mid = (left + right) // 2
            if mid * mid > x:
                right = mid - 1
            elif mid * mid < x:
                left = mid + 1
            else:
                return mid
        return right

# searchInsert

# binarySearch--iterative way ---sorted array

# LeetCode2

class Solutiona(object):
	def searchInsert(self,A,e):
		lo=0
		hi=len(A)-1
		while lo<=hi:
			mid = (lo+hi)//2
			if A[mid]==e:
				return mid
			elif A[mid]>e:
				hi=mid-1
			else:
				lo=mid+1
		return lo

# twoSum

# binarySearch--iterative way -- sorted array

# LeetCode3

class Solutionb(object):
    def twoSum(self,nums,target):
        size = len(nums)
        my_map = {}
        for i in range(size):
            my_map[nums[i]]=i
        for i in range(size):
            if target-nums[i] in my_map:
                return [i+1,my_map[target-nums[i]]+1]


# intersection of two arrays

# Linear time complexity

# LeetCode4

class Solutionc(object):
    def instersection(self,nums1,nums2):
        n1 = set()
        res = set()
        if len(nums1) == 0 or len(nums2) == 0:
            return []
        for num in nums1:
            if num not in n1:
                n1.add(num)


        for num in nums2:
            if num in n1 and num not in res:
                res.add(num)
        return list(res)


# perfect square check

# linear running time complexity------ binary search

# LeetCode5

class Square(object):
    def isPerfect(self,x):
        L = 0
        R = x
        while L <=R :
            mid = (L+R)//2
            if (mid*mid)>x:
                R = mid-1
            elif (mid*mid)<x:
                L = mid+1
            else:
                return mid
        return False


# subsequence checker

# linear time complexity

# LeetCode6

class Sequence(object):
    def isSubsequence(self,s:str,t:str)->bool:
        i=0
        if len(s)==0 and len(t)==0:
            return True
        if len(s)==0:
            return True
        for j in range(len(t)):
            if s[i]==t[j]:
                i +=1
            if len(s)==i:
                return True
        return False


# arrangeCoins

# linear time complexity

# LeetCode7

import math
class Arrange:
    def arrangeCoins(self, n: int) -> int:
        # Given input n we are going to have k stairs and some left over coins
        # Thus the sum of all coins is (k*(k+1))/2 + N = 2*n
        # So we take square root of 2n to approximate k and find it
        k = int(math.sqrt(2*n))
        if k*(k+1) <= 2*n:
            return k
        return k-1



# find the smallest letter greater than target

# LeetCode8

class Greatest:
    def nextGreatestLetter(self, letters, target) :
        if target >= letters[-1]:
            return letters[0]
        sz = len(letters)
        l = 0
        r = sz-1
        while l <= r:
            mid = (l+r)//2
            if letters[mid] <= target:
                l = mid + 1
            else:
                r = mid
        return letters[l]


# binary search with 'yes' indexed it or return -1

# LeetCode9

class Binary:
    def search(self,nums:int,target:int)->int:
        size = len(nums)
        lo=0
        hi=size-1
        while lo<=hi:
            mid = (lo+hi)//2
            if nums[mid]>target:
                hi=mid-1
            elif nums[mid]<target:
                lo=mid+1
            else:
                return nums[mid]
        return -1


# peakfinding in the mountain of an array

# LeetCode

class Mountain(object):
    def peakFinding(self,A):
        L=0
        R=len(A)-1
        while L<=R:
            mid = (L+R)//2
            if (A[mid-1]<A[mid] and A[mid+1]<A[mid]):
                return mid
            elif (A[mid-1]<A[mid] and A[mid]>A[mid+1]):
                L =mid
            else:
               R=mid
        return -1

# g = Mountain()

# g.peakFinding([0,1,0])

# 1. search insert position

# 2. sqrt(x)

# 3. two sum--input array is sorted

# 4. intersection of two arrays

# 5. intersection of  two arrays II

# 6. valid perfect square

# 7. IS Subsequence

# 8. Arranging Coins

# 9. find the smallest letter greater than target

# 10. binary search

# * I    T    A   L   I    A *


#  MITx Python --- 6.0001

#  Linear Time Complexity

# MIT1

class Linear(object):
    def search(self,A,e):
        found =  False # flag


        for i in range(len(A)):
            if e == A[i]:
                found = True
        return found

# MIT2

# nlogn time complexity

class Binary(object):
    """
    Recursion method

    base cases

    convergence

    """
    def bisectionSearch(self,A,e):

        if A==[]:
            return False
        elif len(A)==1:
            return A[0]==e


        else:
            mid = len(A)//2
            if A[mid]>e:
                return self.bisectionSearch(A[:mid],e)
            else:
                return self.bisectionSearch(A[mid:],e)




# MIT3

# Iterative way of multiplication


class Multiter(object):
    def multi(self,a,b):
        '''
        count var

        conditional loop

        reduce the conditional var

        for convergence

        '''
        c=0
        while b>0:
            c+=a
            b-=1
        return c

# MIT4

# Recursive Way of multiplying two nums

class Multrecur(object):
    def multi(self,a,b):
        '''
        base case

        convergence

        recursion

        '''

        if b==1:
            return a
        else:
            return a + self.multi(a,b-1)


# MIT5

# Factorial with iterative way

class Iterates(object):
    def fact(self,n):
        c=1  # counter variable

        while n>0:
            c *=n
            n -=1  # reduction for convergence
        return c


# MIT6

# Factorial with recursive way


class Recur(object):
    def factRecur(self,n):
        '''
        Base case

        Convergence

        Recursion

        '''
        if n==1:
            return 1
        else:
            return n*self.factRecur(n-1)


# MIT7

# Fiboonacci Recursive way


class Fibonacci(object):
    def fibRecursive(self,n):
        if n==0:
            return 0
        if n==1:
            return 1
        else:
            return self.fibRecursive(n-1)+self.fibRecursive(n-2)



# MIT8

# Fibonacci through swapping


class Amico(object):
    def fibSwap(self,n):
        a=0
        b=1
        for i in range(0,n):
            a,b=b,a+b
        return b

# MIT9

# Logarithmic complexity

# inttoStr

class Logcomplexity(object):
    def intToStr(self,i):
        digits = '0123456789'

        if i==0:
            return '0'
        res = ''
        while i>0:
            res = digits[i%10] +res
            i = i//10
        return res


#  MIT10
class QuadraticComplexity(object):
    def isSubset(self,L1, L2):
        for e1 in L1:
            matched = False
            for e2 in L2:
                if e1 == e2:
                    matched = True
                    break
            if not matched:
                return False
        return True




# Bubble Sort

# Selection Sort

# Insertion Sort

# Merge Sort

# Quick Sort

# Counting Sort

# Redix Sort

# Heap Sort

# Bucket Sort

# MIT11

# Bubble sort---comparing pairs of  adjacent

# elements and then swapping their

# positions if they exist in the wrong order.

# The complexity of bubble sort is n**2

class BubbleSort(object):
    def bubbleSort(self,A):
        swap = False
        while not swap:
            swap = True
            for j in range(1,len(A)):
                if A[j-1]>A[j]:
                    swap = False
                    temp = A[j]
                    A[j]=A[j-1]
                    A[j-1]=temp
                    # A[j-1],A[j] = temp,A[j-1]
        return A


# MIT12

# SELECTION SORT

# based on the idea of finding

# the min or max element

# and putting it in its correct

# position in a sorted way

class SelectionSort(object):
    def selectionSort(self,A):
        p = 0
        while p != len(A):
            for i in range(p,len(A)):
                if A[i]<A[p]:
                    A[p],A[i] = A[i],A[p]
            p +=1

        return A

num = 0
while num <= 5:
    print(num)
    num += 1

print("Outside of loop")
print(num)



# program to find maximum in arr[] of size n

def largest(arr):
    n = len(arr)

    # Initialize maximum array
    max = arr[0]

    # Traverse array elements from second
    # and compare every element with current max
    for i in range(1,n):
        if arr[i]>max:
            max = arr[i]
    return max

# driver code
arr = [10,324,45,90,9808]
Ans  = largest(arr)
print(Ans)


def least(arr:list)->int :
    m = len(arr)

    # Initialize the minimum array

    min =  arr[0]

    # Traverse array elments from second
    # and compare every element with current min
    # initialize and update
    for i in range(1,m):
        if arr[i]<min:
            min = arr[i]
    return min

# driver code
arr = [10,324,45,90,9808]
Soln = least(arr)
print(Soln)


# BestTimeToBuyStock

def BestTimeToBuyStock(prices:list)->int:
    if len(prices)==0:
        return 0
    minx = prices[0]  # buy low
    maxy = 0  # sell high
    for i in range(len(prices)):
        if prices[i]>minx:
            maxy = max(maxy,prices[i]-minx)
        else:
            minx = prices[i]
    return maxy

prices=[7,1,5,3,6,4]
print(BestTimeToBuyStock(prices))


# maximum subarray

def max_subarray(arr:list)->int:
    best_sum = 0
    current_sum = 0
    for x in arr:
       current_sum = max(0,current_sum+x)
       best_sum = max(best_sum,current_sum)
    return best_sum


arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_subarray(arr))
