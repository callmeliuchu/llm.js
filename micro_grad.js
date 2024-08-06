class Value{
    constructor(data){
        this.data = data;
        this.grad = 0;
        this.prev = 0;
    }
    backward(child1,op,child2,out){
        if(op == '+'){
            child1.grad = child1.grad + out * 1;
            child2.grad = child2.grad + out * 1;
        }
        if(op == '-'){
            child1.grad = child1.grad + out * 1;
            child2.grad = child2.grad - out * 1;
        }
        if(op == '*'){
            child1.grad = child1.grad + out * child2.data;
            child2.grad = child2.grad + out * child1.data;
        }
        if(op == '/'){
            child1.grad = child1.grad + out / child2.data;
            child2.grad = child2.grad - out * child1.data / child2.data / child2.data;
        }
    }
    operate(op,data){
        res = null;
        other = data;
        if(!data instanceof Value){
            other = Value(data);
        }
        if(op == '+'){
            res = Value(this.data + other.data);
        }
        if(op == '-'){
            res = Value(this.data - other.data);
        }
        if(op == '*'){
            res = Value(this.data * other.data);
        }
        if(op == '/'){
            res = Value(this.data / other.data);
        }
        if(res != null){
            res.prev = [this,op,other];
        }
        return res;
    }
}


function post_travel(obj){
    ans = []
    function f(obj){
        prev = obj.prev
        if(prev != null && prev.length == 3){
            child1 = prev[0];
            op = prev[1];
            child2 = prev[2];
            children = [child1,child2];
            for(o of children){
                f(o);
            }
        }
        ans.push(obj);
    }
    f(obj)
    ans.reverse();
    return ans;
}


function backward(obj){
    ans = post_travel(obj);
    for(o of ans){
        prev = o.prev
        if(prev != null && prev.length == 3){
            child1 = prev[0];
            op = prev[1];
            child2 = prev[2];
            o.backward(child1,op,child2);
        }
    }
}




class Linear{
    constructor(m,n,bias){
        this.m = m;
        this.n = n;
        this.bias = bias;
        this.weights = [];
        for(let i=0;i<this.m;i++){
            let arr = []
            for(let j=0;j<this.n;j++){
                arr.push(Value(1));
            }
            this.weights.push(arr);
        }
    }
    forward(x){
        res = []
        for(let i=0;i<x.lenght;i++){
            sum = Value(0);
            for(let j=0;j<x[i].length;j++){
                sum.operate('+',x[i][j].operate('*',this.weights[j]))
            }
            res.push(sum);
        }
        return res;
    }
}

class MLP{
    constructor(){
        this.linear1 = Linear(5,4,false);
        this.linear2 = Linear(4,1,false);
    }
    forward(x,y){
        x = this.linear1(x)
        x = x.tanh();
        x = this.linear2(x);
        return x;
    }
}


x = []
y = []


mlp = NLP()
parameters = []
for(let i=0;i<10;i++){
    loss = mlp(x,y);
    parameters.zero_grad();
    loss.backward();
    print(loss);
    parameters.step();
}


