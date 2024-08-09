
class Value{
    constructor(data,name){
        this.data = data;
        this.grad = 0;
        this.prev = 0;
        this.required_grad = false;
        this.name = name;
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
        if(op == '**'){
            child1.grad = child2.data * Math.pow(child1.data,child2.data-1) * out;
            child2.grad = Math.pow(child1.data,child2.data) * Math.log(child1.data) * out;
        }
        if(op == 'relu'){
            if(child1.data >= 0){
                child1.grad += out;
            }
        }
    }
    operate(op,data){
        let res = null;
        let other = data;
        if(!(data instanceof Value)){
            other = new Value(data);
        }
        if(op == '+'){
            res = new Value(this.data + other.data);
        }
        if(op == '-'){
            res = new Value(this.data - other.data);
        }
        if(op == '*'){
            res = new Value(this.data * other.data);
        }
        if(op == '/'){
            res = new Value(this.data / other.data);
        }
        if(op == '**'){
            res = new Value(Math.pow(this.data,other.data));
        }
        if(op == 'relu'){
            res = new Value(Math.max(0,this.data));
        }
        if(res != null){
            res.prev = [this,op,other];
        }
        return res;
    }
}
 
// (x1+x2)*x3 + x4 + x1= l

// l_x1 = 1
// l_x4 = 1
// l_x123 = 1
// x123_x1 = x3
// x123_x2 = x1
// l_x123_x1 = 1 * x1

// l_x1 = 1 + 1 * x1

function test2(){
    v1 =new  Value(3,'v1');
    v2 =new Value(2.1,'v2');
    v = v1.operate('**',v2);
    console.log(v.data);
}


function post_travel(obj){
    ans = []
    visited = new Set()
    function f(obj){
        visited.add(obj);
        prev = obj.prev
        if(prev != null && prev.length == 3){
            child1 = prev[0];
            op = prev[1];
            child2 = prev[2];
            children = [child1,child2];
            for(o of children){
                if(!visited.has(o)){
                  f(o);
                }
            }
        }
        ans.push(obj);
    }
    f(obj)
    ans.reverse();
    return ans;
}


function test2(){
    let v1 =new  Value(3,'v1');
    let v2 =new Value(2.1,'v2');
    let v3 = v1.operate('**',v2);
    v3.name = 'v3';
    let v4 = v1.operate('+',v2);
    v4.name = 'v4';
    let v5 = v4.operate('*',v3);
    v5.name = 'v5';
    let res = post_travel(v5);
    for(let o of res){
        console.log(o.name);
    }
}


// test2();

function backward(obj){
    ans = post_travel(obj);
    ans[0].grad = 1.0;
    for(o of ans){
        prev = o.prev;
        if(prev != null && prev.length == 3){
            child1 = prev[0];
            op = prev[1];
            child2 = prev[2];
            o.backward(child1,op,child2,o.grad);
        }
    }
}

function test3(){
    let v1 =new  Value(3,'v1');
    let v2 =new Value(2.1,'v2');
    let v3 = v1.operate('*',v2);
    v3.name = 'v3';
    let v4 = v2.operate('+',v3);
    v4.name = 'v4';
    let v5 = v4.operate('*',v3);
    v5.name = 'v5';
    backward(v5);
    let res = post_travel(v5);
    for(let o of res){
        console.log(o.name,o.grad);
    }
}

class Nerous{
    constructor(){
        this.v1 =new  Value(1,'v1');
    }
    forward(x,y){
        let loss = this.v1.operate('*',x).operate('-',y).operate('**',2);
        return loss;
    }
    parameters(){
        return [this.v1];
    }
}


function zero_grad(params){
    for(let p of params){
        p.grad  = 0;
    }
}

function step_grad(params){
    for(let p of params){
        p.data -= 0.001 * p.grad;
    }
}

function test4(){
    let model = new Nerous();
    let x = 3;
    let y = 20;
    let params = model.parameters();

    for(let i=0;i<1000;i++){
        let loss = model.forward(x,y);
        zero_grad(params);
        backward(loss);
        step_grad(params);
        console.log(x,y,loss.data);
    }
}


class Linear{
    constructor(m,n){
        this.m = m;
        this.n = n;
        this.weights = [];
        for(let i=0;i<this.n;i++){
            let arr = []
            for(let j=0;j<this.m;j++){
                arr.push(new Value(1));
            }
            this.weights.push(arr);
        }
    }
    forward(x){
        // (m,1) ==> (n,1)
        let res = [];
        for(let i=0;i<this.n;i++){
            let sum = new Value(0);
            for(let j=0;j<this.m;j++){
                sum = sum.operate('+',this.weights[i][j].operate('*',x[j]));
            }
            res.push(sum);
        }
        return res;
    }
    parameters(){
        let ans = [];
        for(let i=0;i<this.n;i++){
            for(let j=0;j<this.m;j++){
                ans.push(this.weights[i][j]);
            }
        }
        return ans;
    }
}

function test5(){
    let linear = new Linear(5,4);
    let hidden = linear.forward([1,2,4,6,7]);
    let linear2 = new Linear(4,1);
    console.log(hidden.length,hidden[1].data);
    let output = linear2.forward(hidden);
    console.log(output.length,output[0].data);
}

test5();

function relu(input){
    for(let o of input){
        o.operate('relu',null);
    }
}

class MLP{
    constructor(){
        this.linear1 = new Linear(5,4);
        this.linear2 = new Linear(4,1);
    }
    forward(x,y){
        x = this.linear1.forward(x);
        relu(x);
        x = this.linear2.forward(x);
        let val = x[0];
        let loss = val.operate('-',y).operate('**',2);
        return loss;
    }
    parameters(){
        let ans1 = this.linear1.parameters();
        let ans2 = this.linear2.parameters();
        return ans1.concat(ans2);
    }
}


function test6(){
    let mlp = new MLP();
    let parameters = mlp.parameters();
    let x = [1,2,3,4,5];
    let y = 1;
    for(let i=0;i<10;i++){
        let loss = mlp.forward(x,y);
        zero_grad(parameters);
        backward(loss);
        step_grad(parameters);
        console.log(loss.data);
    }
}




