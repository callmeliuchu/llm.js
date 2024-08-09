
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



function test4(){
    let model = new Nerous();
    let x = 3;
    let y = 20;
    let params = model.parameters();

    for(let i=0;i<1000;i++){
        let loss = model.forward(x,y);
        for(let p of params){
            p.grad  = 0;
        }
        backward(loss);
        for(let p of params){
            p.data -= 0.001 * p.grad;
        }
        console.log(x,y,loss.data);
    }
}

test4()

// test3();


// class Linear{
//     constructor(m,n,bias){
//         this.m = m;
//         this.n = n;
//         this.bias = bias;
//         this.weights = [];
//         for(let i=0;i<this.m;i++){
//             let arr = []
//             for(let j=0;j<this.n;j++){
//                 arr.push(Value(1));
//             }
//             this.weights.push(arr);
//         }
//     }
//     forward(x){
//         res = []
//         for(let i=0;i<x.lenght;i++){
//             sum = Value(0);
//             for(let j=0;j<x[i].length;j++){
//                 sum.operate('+',x[i][j].operate('*',this.weights[j]))
//             }
//             res.push(sum);
//         }
//         return res;
//     }
// }

// class MLP{
//     constructor(){
//         this.linear1 = Linear(5,4,false);
//         this.linear2 = Linear(4,1,false);
//     }
//     forward(x,y){
//         x = this.linear1(x)
//         x = x.tanh();
//         x = this.linear2(x);
//         return x;
//     }
// }


// x = []
// y = []


// mlp = NLP()
// parameters = []
// for(let i=0;i<10;i++){
//     loss = mlp(x,y);
//     parameters.zero_grad();
//     loss.backward();
//     print(loss);
//     parameters.step();
// }


