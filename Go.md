# 小根堆示例：

```go
func kSum(nums []int, k int) int64 {
    n := len(nums)
    sum := 0
    for i, x := range nums { 
        if x >= 0 {
            sum += x
        } else {
            nums[i] = -x
        }
    }
    slices.Sort(nums)

    h := hp{{0, 0}} // 空子序列
    for ; k > 1; k-- {
        p := heap.Pop(&h).(pair)
        i := p.i
        if i < n {
            // 在子序列的末尾添加 nums[i]
            heap.Push(&h, pair{p.sum + nums[i], i + 1}) // 下一个添加/替换的元素下标为 i+1
            if i > 0 { // 替换子序列的末尾元素为 nums[i]
                heap.Push(&h, pair{p.sum + nums[i] - nums[i-1], i + 1})
            }
        }
    }
    return int64(sum - h[0].sum)
}

type pair struct{ sum, i int }
type hp []pair
func (h hp) Len() int            { return len(h) }
func (h hp) Less(i, j int) bool  { return h[i].sum < h[j].sum }
func (h hp) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(v any)         { *h = append(*h, v.(pair)) }
func (h *hp) Pop() any           { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }

```

 ```go
 func minOperations(nums []int, k int) int {
     ans := 0
     h := &hp{nums}
     heap.Init(h)
     for h.IntSlice[0] < k {
         x := heap.Pop(h).(int)
         h.IntSlice[0] += x * 2
         heap.Fix(h,0)
         ans++
     }
     return ans
 }
 
 type hp struct{sort.IntSlice}
 func (hp) Push(any) {}
 //func (s *hheap) Push(v interface{}) {
 	//s.IntSlice = append(s.IntSlice, v.(int))
 //}
 func (h *hp) Pop() any {
     a := h.IntSlice
     v := a[len(a)-1]
     h.IntSlice = a[:len(a)-1]
     return v
 }
 
 ```

 

# 图论

```go
func numIslands(grid [][]byte) int {
    res := 0
    for i := 0; i < len(grid); i++ {
        for j := 0; j < len(grid[i]); j++ {
            if grid[i][j] == '1' {
                res++
                dfs(grid,i,j)
            }
        }
    }
    return res
}

func dfs(grid [][]byte,r,c int) {
    h,w := len(grid),len(grid[0])
    if r < 0 || r >= h || c < 0 || c >= w {
        return
    }
    if grid[r][c] == '0' {
        return
    }
    grid[r][c] = '0'
    dfs(grid,r-1,c)
    dfs(grid,r+1,c)
    dfs(grid,r,c-1)
    dfs(grid,r,c+1)
} 

```

## java版

```java
void dfs(int[][] grid, int r, int c) {
    // 判断 base case
    if (!inArea(grid, r, c)) {
        return;
    }
    // 如果这个格子不是岛屿，直接返回
    if (grid[r][c] != 1) {
        return;
    }
    grid[r][c] = 2; // 将格子标记为「已遍历过」
    
    // 访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
boolean inArea(int[][] grid, int r, int c) {
    return 0 <= r && r < grid.length 
        	&& 0 <= c && c < grid[0].length;
}

```



 

# 快速幂

 ```go
 func myPow(x float64, n int) float64 {
     if n >= 0 {
         return qPow(x,n)
     }else {
         return 1.0/qPow(x,-n)
     }
 }
 
 func qPow(x float64,n int) float64 {
     res := 1.0
     for n > 0 {
         if n%2 == 1 {
             res *= x
         }
         x *= x
         n /= 2
     }
     return res
 }
 
 ```



```go
func pow(x, n int) int {
    res := 1
    for ; n > 0; n /= 2 {
        if n%2 > 0 {
            res = res * x
        }
        x = x * x
    }
    return res
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/ways-to-express-an-integer-as-sum-of-powers/solutions/2354970/0-1-bei-bao-mo-ban-ti-by-endlesscheng-ap09/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```





 

 

# 2642邻接矩阵建图 + 朴素 Dijkstra  （适合稠密图）

```go
const inf = math.MaxInt / 2 // 防止更新最短路时加法溢出

type Graph [][]int

func Constructor(n int, edges [][]int) Graph {
    g := make([][]int, n) // 邻接矩阵
    for i := range g {
        g[i] = make([]int, n)
        for j := range g[i] {
            g[i][j] = inf // 初始化为无穷大，表示 i 到 j 没有边
        }
    }
    for _, e := range edges {
        g[e[0]][e[1]] = e[2] // 添加一条边（题目保证没有重边）
    }
    return g
}

func (g Graph) AddEdge(e []int) {
    g[e[0]][e[1]] = e[2] // 添加一条边（题目保证这条边之前不存在）
}

func (g Graph) ShortestPath(start, end int) int {
    n := len(g)
    dis := make([]int, n) // 从 start 出发，到各个点的最短路，如果不存在则为无穷大
    for i := range dis {
        dis[i] = inf
    }
    dis[start] = 0
    vis := make([]bool, n)
    for {
        x := -1
        for i, b := range vis {
            if !b && (x < 0 || dis[i] < dis[x]) {
                x = i
            }
        }
        if x < 0 || dis[x] == inf { // 所有从 start 能到达的点都被更新了
            return -1
        }
        if x == end { // 找到终点，提前退出
            return dis[x]
        }
        vis[x] = true // 标记，在后续的循环中无需反复更新 x 到其余点的最短路长度
        for y, w := range g[x] {
            dis[y] = min(dis[y], dis[x]+w) // 更新最短路长度
        }
    }
}

```



 

 

## 邻接表建图 + 堆优化 Dijkstra （适合稀疏图）

```go
type Graph [][]pair
func Constructor(n int, edges [][]int) Graph {
    g := make(Graph, n) // 邻接表
    for _, e := range edges {
        g[e[0]] = append(g[e[0]], pair{e[1], e[2]})
    }
    return g
}

func (g Graph) AddEdge(e []int) {
    g[e[0]] = append(g[e[0]], pair{e[1], e[2]})
}

func (g Graph) ShortestPath(start, end int) int {
    // dis[i] 表示从起点 start 出发，到节点 i 的最短路长度
    dis := make([]int, len(g))
    for i := range dis {
        dis[i] = math.MaxInt
    }
    dis[start] = 0
    h := hp{{start, 0}}
    for len(h) > 0 {
        p := heap.Pop(&h).(pair)
        x, d := p.x, p.d
        if x == end { // 计算出从起点到终点的最短路长度
            return d
        }
        if d > dis[x] { // x 之前出堆过，无需更新邻居的最短路
            continue
        }
        for _, e := range g[x] {
            y, w := e.x, e.d
            newD := d + w
            if newD < dis[y] {
                dis[y] = newD // 更新最短路长度
                heap.Push(&h, pair{y, newD})
            }
        }
    }
    return -1 // 无法到达终点
}

type pair struct{ x, d int }
type hp []pair
func (h hp) Len() int           { return len(h) }
func (h hp) Less(i, j int) bool { return h[i].d < h[j].d }
func (h hp) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(v any)        { *h = append(*h, v.(pair)) }
func (h *hp) Pop() (v any)      { a := *h; *h, v = a[:len(a)-1], a[len(a)-1]; return }

```



# 数位DP(2719)



```go
func count(num1, num2 string, minSum, maxSum int) int {
    const mod = 1_000_000_007
    n := len(num2)
    num1 = strings.Repeat("0", n-len(num1)) + num1 // 补前导零，和 num2 对齐

    memo := make([][]int, n)
    for i := range memo {
        memo[i] = make([]int, min(9*n, maxSum)+1)
        for j := range memo[i] {
            memo[i][j] = -1
        }
    }
    var dfs func(int, int, bool, bool) int
    dfs = func(i, sum int, limitLow, limitHigh bool) (res int) {
        if sum > maxSum { // 非法
            return
        }
        if i == n {
            if sum >= minSum { // 合法
                return 1
            }
            return
        }

        if !limitLow && !limitHigh {
            p := &memo[i][sum]
            if *p >= 0 {
                return *p
            }
            defer func() { *p = res }()
        }

        lo := 0
        if limitLow {
            lo = int(num1[i] - '0')
        }
        hi := 9
        if limitHigh {
            hi = int(num2[i] - '0')
        }

        for d := lo; d <= hi; d++ { // 枚举当前数位填 d
            res = (res + dfs(i+1, sum+d, limitLow && d == lo, limitHigh && d == hi)) % mod
        }
        return
    }
    return dfs(0, 0, true, true)
}


作者：灵茶山艾府
链接：https://leetcode.cn/problems/count-of-integers/solutions/2296043/shu-wei-dp-tong-yong-mo-ban-pythonjavacg-9tuc/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```



 

## 902

```go
func atMostNGivenDigitSet(digits []string, n int) int {
    s := strconv.Itoa(n)
    m := len(s)
    dp := make([]int, m)
    for i := range dp {
        dp[i] = -1 // dp[i] = -1 表示 i 这个状态还没被计算出来
    }
    var f func(int, bool, bool) int
    f = func(i int, isLimit, isNum bool) (res int) {
        if i == m {
            if isNum { // 如果填了数字，则为 1 种合法方案
                return 1
            }
            return
        }
        if !isLimit && isNum { // 在不受到任何约束的情况下，返回记录的结果，避免重复运算
            dv := &dp[i]
            if *dv >= 0 {
                return *dv
            }
            defer func() { *dv = res }()
        }
        if !isNum { // 前面不填数字，那么可以跳过当前数位，也不填数字
            // isLimit 改为 false，因为没有填数字，位数都比 n 要短，自然不会受到 n 的约束
            // isNum 仍然为 false，因为没有填任何数字
            res += f(i+1, false, false)
        }
        // 根据是否受到约束，决定可以填的数字的上限
        up := byte('9')
        if isLimit {
            up = s[i]
        }
        // 注意：对于一般的题目而言，如果此时 isNum 为 false，则必须从 1 开始枚举，由于本题 digits 没有 0，所以无需处理这种情况
        for _, d := range digits { // 枚举要填入的数字 d
            if d[0] > up { // d 超过上限，由于 digits 是有序的，后面的 d 都会超过上限，故退出循环
                break
            }
            // isLimit：如果当前受到 n 的约束，且填的数字等于上限，那么后面仍然会受到 n 的约束
            // isNum 为 true，因为填了数字
            res += f(i+1, isLimit && d[0] == up, true)
        }
        return
    }
    return f(0, true, false)
}

作者：灵茶山艾府
链接：https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/solutions/1900101/shu-wei-dp-tong-yong-mo-ban-xiang-xi-zhu-e5dg/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```



 

# 字典树

```go
// LC 3093
func stringIndices(wordsContainer, wordsQuery []string) []int {
    type node struct {
        son     [26]*node
        minL, i int
    }
    root := &node{minL: math.MaxInt}

    for idx, s := range wordsContainer {
        l := len(s)
        cur := root
        if l < cur.minL {
            cur.minL, cur.i = l, idx
        }
        for i := len(s) - 1; i >= 0; i-- {
            b := s[i] - 'a'
            if cur.son[b] == nil {
                cur.son[b] = &node{minL: math.MaxInt}
            }
            cur = cur.son[b]
            if l < cur.minL {
                cur.minL, cur.i = l, idx
            }
        }
    }

    ans := make([]int, len(wordsQuery))
    for idx, s := range wordsQuery {
        cur := root
        for i := len(s) - 1; i >= 0 && cur.son[s[i]-'a'] != nil; i-- {
            cur = cur.son[s[i]-'a']
        }
        ans[idx] = cur.i
    }
    return ans
}


作者：灵茶山艾府
链接：https://leetcode.cn/problems/longest-common-suffix-queries/solutions/2704763/zi-dian-shu-wei-hu-zui-duan-chang-du-he-r3h3j/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

~~~go
// LC 140
type Trie struct {
    children [26]*Trie
    isEnd bool
}


func wordBreak(s string, wordDict []string) []string {
    res := make([]string,0)
    root := &Trie{}
    for _,word := range wordDict {
        node := root
        for i := 0; i < len(word); i++ {
            index := word[i] - 'a'
            if node.children[index] == nil {
                node.children[index] = &Trie{}
            }
            node = node.children[index]
        }
        node.isEnd = true       
    }
    
    var dfs func(int,[]string,int)
    dfs = func(start int,curr []string,l int) {
        if l == len(s) {
            res = append(res,wrap(curr))
            return
        }
        node := root
        for i := start; i < len(s) ; i++ {
            index := s[i]-'a'
            if node.children[index] != nil {
                node = node.children[index]
                if node.isEnd {
                    cl := i-start+1
                    curr = append(curr,s[start:i+1])
                    l += cl
                    dfs(i+1,curr,l)
                    l -= cl
                    curr = curr[0:len(curr)-1]
                }
            } else {
                break
            }
        }
    }
    dfs(0,[]string{},0)
    return res
}

func wrap(curr []string) string {
    res := ""
    for _,v := range curr {
        res += v
        res += " "
    }
    return res[0:len(res)-1]
}
~~~



# 树上倍增 寻找LCA

```go
type TreeAncestor struct {
    depth []int
    pa    [][]int
}

func Constructor(edges [][]int) *TreeAncestor {
    n := len(edges) + 1
    m := bits.Len(uint(n))
    g := make([][]int, n)
    for _, e := range edges {
        x, y := e[0], e[1] // 节点编号从 0 开始
        g[x] = append(g[x], y)
        g[y] = append(g[y], x)
    }

    depth := make([]int, n)
    pa := make([][]int, n)
    var dfs func(int, int)
    dfs = func(x, fa int) {
        pa[x] = make([]int, m)
        pa[x][0] = fa
        for _, y := range g[x] {
            if y != fa {
                depth[y] = depth[x] + 1
                dfs(y, x)
            }
        }
    }
    dfs(0, -1)

    for i := 0; i < m-1; i++ {
        for x := 0; x < n; x++ {
            if p := pa[x][i]; p != -1 {
                pa[x][i+1] = pa[p][i]
            } else {
                pa[x][i+1] = -1
            }
        }
    }
    return &TreeAncestor{depth, pa}
}

func (t *TreeAncestor) GetKthAncestor(node, k int) int {
    for ; k > 0; k &= k - 1 {
        node = t.pa[node][bits.TrailingZeros(uint(k))]
    }
    return node
}

// 返回 x 和 y 的最近公共祖先（节点编号从 0 开始）
func (t *TreeAncestor) GetLCA(x, y int) int {
    if t.depth[x] > t.depth[y] {
        x, y = y, x
    }
    y = t.GetKthAncestor(y, t.depth[y]-t.depth[x]) // 使 y 和 x 在同一深度
    if y == x {
        return x
    }
    for i := len(t.pa[x]) - 1; i >= 0; i-- {
        px, py := t.pa[x][i], t.pa[y][i]
        if px != py {
            x, y = px, py // 同时上跳 2^i 步
        }
    }
    return t.pa[x][0]
}


作者：灵茶山艾府
链接：https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/solutions/2305895/mo-ban-jiang-jie-shu-shang-bei-zeng-suan-v3rw/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```





#  小技巧

## 排序去重

```go
slices.Sort(nums) //需要先排序
a := slices.Compact(nums) // 原地去重

//按需排序
func demoSortSlice(){
    a := []int{6,3,9,8,1,2,5,7}
    sort.Slice(a, func(i, j int) bool {
        return a[i]>a[j]
    })
    fmt.Println(a)
    //[9 8 7 6 5 3 2 1]
}

```

## 最大公约数

```go
func gcd(a,b int)int {
    for a != 0 {
        a,b = b%a,a
    }
    return b
}

```

 

## 数组找最小

```go
slices.Min(initial)
```

## 向上取整

<span style="font-size: 25px">$\lceil \frac{a}{b} \rceil$ = $\lfloor \frac{a+b-1}{b} \rfloor$</span>

## 数组克隆

```go
ans = append(ans,slices.Clone(path))
```

## 素数（埃氏筛）

```go
func countPrimes(n int) (cnt int) {
    isPrime := make([]bool, n)
    for i := range isPrime {
        isPrime[i] = true
    }
    for i := 2; i < n; i++ {
        if isPrime[i] {
            cnt++
            for j := 2 * i; j < n; j += i {
                isPrime[j] = false
            }
        }
    }
    return
}

```

## 自定义排序

~~~go
sort.Slice(a,func(i,j int) bool {
        return a[i].dis < a[j].dis
    })
~~~



# 树状数组、线段树

此处可以再总结一下（加粗字体为最佳方案）：

数组不变，区间查询：**前缀和**、树状数组、线段树；
数组单点修改，区间查询：**树状数组**、线段树；
数组区间修改，单点查询：**差分**、线段树；
数组区间修改，区间查询：**线段树**。

## LC 307

 ```go
 // LC307
 type NumArray struct {
     nums []int
     tree []int
 }
 
 func Constructor(nums []int) NumArray {
     tree := make([]int, len(nums)+1)
     for i, x := range nums {
         i++
         tree[i] += x
         if nxt := i + i&-i; nxt < len(tree) {
             tree[nxt] += tree[i]
         }
     }
     return NumArray{nums, tree}
 }
 
 func (a *NumArray) Update(index, val int) {
     delta := val - a.nums[index]
     a.nums[index] = val
     for i := index + 1; i < len(a.tree); i += i & -i {
         a.tree[i] += delta
     }
 }
 
 func (a *NumArray) prefixSum(i int) (s int) {
     for ; i > 0; i &= i - 1 { // i -= i & -i 的另一种写法
         s += a.tree[i]
     }
     return
 }
 
 func (a *NumArray) SumRange(left, right int) int {
     return a.prefixSum(right+1) - a.prefixSum(left)
 }
 
 ```

## 树状数组示例（LC 1395）

```go
func numTeams(rating []int) (ans int) {
    n := len(rating)
    trees := make([]int, 1e5+1)
    less := make([]int, n)
    for i, r := range rating {
        for x := r; x <= 1e5; x += x & -x {
            trees[x]++
        }
        tmp := 0
        for x := r-1; x > 0; x &= x-1 {
            tmp += trees[x]
        }
        less[i] = tmp
    }
    trees = make([]int, 1e5+1)
    for i := n-1; i >= 0; i-- {
        for x := rating[i]; x <= 1e5; x += x & -x {
            trees[x]++
        }
        tmp := 0
        for x := rating[i]-1; x > 0; x &= x-1 {
            tmp += trees[x]
        }
        // 小于 * 大于 + 大于 * 小于
        ans += less[i] * (n-i-tmp-1) + (i-less[i])*tmp
    }
    return ans
}

```

## 树状数组的应用 3072

~~~go
type fenwick []int
func (f fenwick) add(i int) {
    for ;i < len(f); i += i&-i {
        f[i]++
    }
}

func (f fenwick) pre(i int) (res int) {
    for ; i > 0; i &= i-1 {
        res += f[i]
    }
    return
}



func resultArray(nums []int) []int {
    sorted := slices.Clone(nums)
    slices.Sort(sorted)
    sorted = slices.Compact(sorted)
    m := len(sorted)

    a := []int{nums[0]}
    b := []int{nums[1]}
    t1 := make(fenwick,m+1)
    t2 := make(fenwick,m+1)
    // 注意要加一这样可以避开0，因为lowerbit里面是不会有0d
    t1.add(sort.SearchInts(sorted,nums[0])+1)
    t2.add(sort.SearchInts(sorted,nums[1])+1)
    for _,x := range nums[2:] {
        v := sort.SearchInts(sorted,x)+1
        gc1 := len(a) - t1.pre(v)
        gc2 := len(b) - t2.pre(v)
        if gc1 > gc2 || gc1 == gc2 && len(a) <= len(b) {
            a = append(a,x)
            t1.add(v)
        } else {
            b = append(b,x)
            t2.add(v)
        }
    }
    return append(a,b...)

}
~~~

## 线段树

单点修改，区间查询最大值

~~~go
// LC 2407
type seg []struct{ l, r, max int }

func (t seg) build(o, l, r int) {
	t[o].l, t[o].r = l, r
	if l == r {
		return
	}
	m := (l + r) >> 1
	t.build(o<<1, l, m)
	t.build(o<<1|1, m+1, r)
}
// o代表节点，i代表数值，val代表最长符合条件的递增数组长度
func (t seg) modify(o, i, val int) {
	if t[o].l == t[o].r {
		t[o].max = val
		return
	}
	m := (t[o].l + t[o].r) >> 1
	if i <= m {
		t.modify(o<<1, i, val)
	} else {
		t.modify(o<<1|1, i, val)
	}
	t[o].max = max(t[o<<1].max, t[o<<1|1].max)
}

// 返回区间 [l,r] 内的最大值
func (t seg) query(o, l, r int) int {
	if l <= t[o].l && t[o].r <= r {
		return t[o].max
	}
	m := (t[o].l + t[o].r) >> 1
	if r <= m {
		return t.query(o<<1, l, r)
	}
	if m < l {
		return t.query(o<<1|1, l, r)
	}
	return max(t.query(o<<1, l, r), t.query(o<<1|1, l, r))
}

func lengthOfLIS(nums []int, k int) int {
	mx := 0
	for _, x := range nums {
		mx = max(mx, x)
	}
    // 保守起见开4n，其实可以更小
	t := make(seg, mx*4)
    // 用值域来建线段树，这样可以更好地方便查询符合条件的前一个状态
	t.build(1, 1, mx)
	for _, x := range nums {
		if x == 1 {
			t.modify(1, 1, 1)
		} else {
			t.modify(1, x, 1+t.query(1, max(x-k, 1), x-1))
		}
	}
	return t[1].max
}

func max(a, b int) int {
	if b > a {
		return b
	}
	return a
}
~~~

利用区间最大值来二分查找大于当签数的第一个数

~~~go
// LC 2940
type seg []int

func (t seg) build(a []int,o,l,r int) {
    if l == r {
        t[o] = a[l]
        return
    }
    m := (l+r)>>1
    t.build(a,o<<1,l,m)
    t.build(a,o<<1|1,m+1,r)
    t[o] = max(t[o<<1],t[o<<1|1])
}

func (t seg) query(o,l,r,L,v int) int {
    if t[o] <= v {
        return -1
    }
    if l == r {
        return l
    }
    m := (l+r) >> 1
    if L <= m {
        pos := t.query(o<<1,l,m,L,v)
        if pos >= 0 {
            return pos
        }
    }
    return t.query(o<<1|1,m+1,r,L,v)
}

func leftmostBuildingQueries(heights []int, queries [][]int) []int {
    n := len(heights)
    t := make(seg,2<<bits.Len(uint(n)))
    t.build(heights,1,0,n-1)
    ans := make([]int,len(queries))
    for i,q := range queries {
        a,b := q[0],q[1]
        if a > b {
            a,b = b,a
        }
        if a == b || heights[a] < heights[b] {
            ans[i] = b
        } else {
            ans[i] = t.query(1,0,n-1,b+1,heights[a])
        }
    }
    return ans
}
~~~

## 带懒标记的线段树（区间更新）

~~~go
// LC 2569
type seg []struct {
    l,r,cnt1 int
    flip bool
}

func (t seg) maintain(o int) {
    t[o].cnt1 = t[o<<1].cnt1+t[o<<1|1].cnt1
}

func (t seg) build(a []int,o,l,r int) {
    t[o].l,t[o].r = l,r
    if l == r {
        t[o].cnt1 = a[l-1]
        return
    }
    m := (l+r) >> 1
    t.build(a,o<<1,l,m)
    t.build(a,o<<1|1,m+1,r)
    t.maintain(o)
}

func (t seg) do(i int) {
    o := &t[i]
    o.cnt1 = o.r-o.l+1-o.cnt1
    o.flip = !o.flip
}

func (t seg) spread(o int) {
    if t[o].flip {
        t.do(o<<1)
        t.do(o<<1|1)
        t[o].flip = false
    }
}

func (t seg) update(o,l,r int) {
    if l <= t[o].l && t[o].r <= r {
        t.do(o)
        return
    }
    t.spread(o)
    m := (t[o].l+t[o].r)>>1
    if l <= m {
        t.update(o<<1,l,r)
    }
    if m < r {
        t.update(o<<1|1,l,r)
    }
    t.maintain(o)
}

func handleQuery(nums1 []int, nums2 []int, queries [][]int) (ans []int64) {
    sum := 0
    for _,x := range nums2 {
        sum += x
    }
    n := len(nums1)
    t := make(seg,2<<bits.Len(uint(n)))
    t.build(nums1,1,1,n)
    for _,q := range queries {
        if q[0] == 1 {
            t.update(1,q[1]+1,q[2]+1)
        } else if q[0] == 2 {
            sum += q[1]*t[1].cnt1
        } else {
            ans = append(ans,int64(sum))
        }
    }
    return 
}
~~~



# 回文串

存一个二维数组，g [i] [j]表示字符串s从下标i到下标j是否为回文串。

~~~go
g := make([][]bool,n)
for i := range g {
    g[i] = make([]bool,n)
    for j := range g[i] {
        g[i][j] = true
    }
}

for i := n-1; i >= 0; i-- {
    for j := i+1; j < n; j++ {
        g[i][j] = s[i] == s[j] && g[i+1][j-1]
    }
}
~~~

## 中心扩展法

~~~go
func countSubstrings(s string) int {
    n := len(s)
    ans := 0
    for i := 0; i < 2 * n - 1; i++ {
        l, r := i / 2, i / 2 + i % 2
        for l >= 0 && r < n && s[l] == s[r] {
            l--
            r++
            ans++
        }
    }
    return ans
}
~~~



# 字符串自定义分割

~~~go
ff := func(r rune) bool { return !unicode.IsLetter(r) }
// split contents into an array of words.
words := strings.FieldsFunc(contents, ff)
~~~



# 按位或最大的最小子数组长度 2411（模板）

~~~go
func smallestSubarrays(nums []int) []int {
	n := len(nums)
	ans := make([]int, n)
	type pair struct{ or, i int }
	ors := []pair{} // 按位或的值 + 对应子数组的右端点的最小值
	for i := n - 1; i >= 0; i-- {
		num := nums[i]
		ors = append(ors, pair{0, i})
		ors[0].or |= num
		k := 0
		for _, p := range ors[1:] {
			p.or |= num
			if ors[k].or == p.or {
				ors[k].i = p.i // 合并相同值，下标取最小的
			} else {
				k++
				ors[k] = p
			}
		}
		ors = ors[:k+1]
        // 本题只用到了 ors[0]，如果题目改成任意给定数字，可以在 ors 中查找
		ans[i] = ors[0].i - i + 1
	}
	return ans
}
~~~

# 递推式求$C_{n}^{k}$

使用的是$C_{n}^{k} = C_{n-1}^{k} + C_{n-1}^{k-1}$

~~~go
// LC2400
    f := make([][]int,k+1)
    for i := range f {
        f[i] = make([]int,k+1)
    }
    for i := range k+1 {
        f[i][0] = 1
        for j := 1; j <= i; j++ {
            f[i][j] = (f[i-1][j]+f[i-1][j-1]) % mod
        }
    }
~~~

# 反向dp 

由于状态转移需要从后往前转移 例如转移方程为 f[i] = f[i+1]+1 如果是正序dp那么遍历到f[i]的时候f[i+1]还没有遍历到，无法转移过来，所以我们需要反向dp，即从后往前dp，这样当遍历到f[i]的时候f[i+1]已经遍历过并且是我们需要的状态。

~~~go
// 示例 LC983
func mincostTickets(days []int, costs []int) int {
    n := days[len(days)-1]
    m := len(days)
    f := make([]int,n+2)
    p := m-1
    for i := n; i > 0; i-- {
        f[i] = f[i+1]
        if p >= 0 && days[p] == i {
            c1,c2,c3 := costs[0],costs[1],costs[2]
            if n - i >= 1 {
                c1 += f[i+1]
            }
            if n - i >= 7 {
                c2 += f[i+7]
            }
            if n - i >= 30 {
                c3 += f[i+30]
            }
            f[i] = min(c1,c2,c3)
            p--
        }
    }
    //fmt.Println(f)
    return f[1]
}
~~~



# 状态压缩DP

~~~go
// LC 2305
func distributeCookies(a []int, k int) int {
	n := 1 << len(a)
	sum := make([]int, n)
    // 计算各个子集的和
	for i, v := range a {
		for j, bit := 0, 1<<i; j < bit; j++ {
			sum[bit|j] = sum[j] + v
		}
	}

	f := append([]int{}, sum...)
	for i := 1; i < k; i++ {
		for j := n - 1; j > 0; j-- {
            // 快速枚举所有j的子集
			for s := j; s > 0; s = (s - 1) & j {
				f[j] = min(f[j], max(f[j^s], sum[s]))
			}
		}
	}
	return f[n-1]
}

~~~

~~~go
// LC 526
func countArrangement(n int) int {
    s := 1<<n
    f := make([]int,s)
    f[0] = 1
    for i := range s {
        lens := bits.OnesCount(uint(i))
        for j := range n {
            if i >> j & 1 == 1 && ( (lens% (j+1) == 0) || (j+1) % lens == 0 ) {
                f[i] += f[i^(1<<j)]
            }
        }
    }
    return f[s-1]
}
~~~

# KMP算法

~~~go
// LC 3036
func countMatchingSubarrays(nums []int, pattern []int) int {
    m := len(pattern)
    pi := make([]int,m)
    cnt := 0
    for i := 1; i < m; i++ {
        v := pattern[i]
        // 当前串匹配不上就从当前串再找最长前后缀，递归
        // cnt是当前比较的字符的位置，如果要回退的话得看之前一个字符的pi数组中的值
        // pi中记录的是cnt，即下一个要比较的地方
        for cnt > 0 && v != pattern[cnt] {
            cnt = pi[cnt-1]
        }
        if v == pattern[cnt] {
            cnt++
        }
        pi[i] = cnt
    }
    cnt = 0
    ans := 0
    for i := 0; i < len(nums)-1; i++ {
        s := cmp.Compare(nums[i+1],nums[i])
        for cnt > 0 && s != pattern[cnt] {
            cnt = pi[cnt-1]
        }
        if s == pattern[cnt] {
            cnt++
        }
        if cnt == m {
            ans++
            cnt = pi[cnt-1]
        }
    }
    return ans
}
~~~





# 并查集

~~~go
type UnionFind struct {
	Fa     []int
	Groups int // 连通分量个数
}

func NewUnionFind(n int) UnionFind {
	fa := make([]int, n) // n+1
	for i := range fa {
		fa[i] = i
	}
	return UnionFind{fa, n}
}
func (u UnionFind) Find(x int) int {
	if u.Fa[x] != x {
		u.Fa[x] = u.Find(u.Fa[x])
	}
	return u.Fa[x]
}
// newRoot = -1 表示未发生合并
func (u *UnionFind) Merge(from, to int) (newRoot int) {
	x, y := u.Find(from), u.Find(to)
	if x == y {
		return -1
	}
	u.Fa[x] = y
	u.Groups--
	return y
}

func (u UnionFind) Same(x, y int) bool {
	return u.Find(x) == u.Find(y)
}
~~~

~~~go
// LC 947
type unionFind struct {
	parent map[int]int
	cnt    int
}

func NewUnionFind() *unionFind {
	return &unionFind{
		parent: make(map[int]int),
	}
}

func (u *unionFind) getCount() int {
	return u.cnt
}

func (u *unionFind) find(x int) int {
	if _, ok := u.parent[x]; !ok {
		u.parent[x] = x
		// 并查集集中新加入一个结点，结点的父亲结点是它自己，所以连通分量的总数 +1
		u.cnt++
	}

	if x != u.parent[x] {
		u.parent[x] = u.find(u.parent[x])
	}
	return u.parent[x]
}

func (u *unionFind) union(x, y int) {
	rootX, rootY := u.find(x), u.find(y)
	if rootX == rootY {
		return
	}
	u.parent[rootX] = rootY
	// 两个连通分量合并成为一个，连通分量的总数 -1
	u.cnt--
}

func removeStones(stones [][]int) int {
	unionf := NewUnionFind()
	for _, s := range stones {
		unionf.union(s[0]+114514, s[1])//增加114514偏移量，确保石头的坐标在并查集中作为节点的标识符时不会重复
	}
	return len(stones) - unionf.getCount()//石头的总数减去连通分量的数量，即最少需要移除的石头数
}
~~~



~~~go
type unionFind struct {
	parent map[int]int
}

func NewUnionFind() *unionFind {
	return &unionFind{
		parent: make(map[int]int),
	}
}

func (u *unionFind) find(x int) int {
	if _, ok := u.parent[x]; !ok {
		u.parent[x] = x
	}

	if x != u.parent[x] {
		u.parent[x] = u.find(u.parent[x])
	}
	return u.parent[x]
}

func (u *unionFind) union(x, y int) {
	rootX, rootY := u.find(x), u.find(y)
	if rootX == rootY {
		return
	}
	u.parent[rootX] = rootY
}
~~~



# 三路快排

~~~go
func findKthLargest(nums []int, k int) int {
    return quickselect(nums, k)
}

func quickselect(arr []int, k int) int{
    
    pivot := arr[len(arr)/2]
    left := []int{}
    right := []int{}
    middle := []int{}
    
    for _, num := range arr {
        if num > pivot {
            left = append(left, num)
        } else if num < pivot {
            right = append(right, num)
        } else {
            middle = append(middle, num)
        }
    }
    if k <= len(left) {
        return quickselect(left,k)
    }
    if len(left)+len(middle) < k {
        return quickselect(right,k-len(left)-len(middle))
    }
    return pivot
}
~~~



# 离线堆

~~~go
// LC 2940
func leftmostBuildingQueries(heights []int, queries [][]int) []int {
    ans := make([]int, len(queries))
    for i := range ans {
        ans[i] = -1
    }
    qs := make([][]pair, len(heights))
    for i, q := range queries {
        a, b := q[0], q[1]
        if a > b {
            a, b = b, a // 保证 a <= b
        }
        if a == b || heights[a] < heights[b] {
            ans[i] = b // a 直接跳到 b
        } else {
            qs[b] = append(qs[b], pair{heights[a], i}) // 离线询问
        }
    }

    h := hp{}
    for i, x := range heights {
        for h.Len() > 0 && h[0].h < x {
            // 堆顶的 heights[a] 可以跳到 heights[i]
            ans[heap.Pop(&h).(pair).i] = i
        }
        for _, p := range qs[i] {
            heap.Push(&h, p) // 后面再回答
        }
    }
    return ans
}

type pair struct{ h, i int }
type hp []pair
func (h hp) Len() int           { return len(h) }
func (h hp) Less(i, j int) bool { return h[i].h < h[j].h }
func (h hp) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(v any)        { *h = append(*h, v.(pair)) }
func (h *hp) Pop() any          { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }
~~~

