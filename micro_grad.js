

function sigmoid(x){
    return 1/(1+Math.exp(-x));
 }
 function sigmoid_(x){
     return (1-sigmoid(x))*sigmoid(x);
 }
 
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
         if(op == 'sigmoid'){
             child1.grad = child1.grad + out*sigmoid_(child1.data);
         }
         if(op == 'relu'){
             if(child1.data >= 0){
                 child1.grad = child1.grad + out;
             }
         }
         if(op == 'log'){
             child1.grad = child1.grad + out / child1.data;
         }
         if(op == 'exp'){
             child1.grad = child1.grad + out * Math.exp(child1.data);
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
             if(this.data > 0){
                 res = new Value(this.data);
             }else{
                 res = new Value(0);
             }
         }
         if(op == 'sigmoid'){
             res = new Value(sigmoid(this.data));
         }
         if(op == 'log'){
             res = new Value(Math.log(this.data));
         }
         if(op == 'exp'){
             res = new Value(Math.exp(this.data));
         }
         if(res != null){
             res.prev = [this,op,other];
         }
         return res;
     }
 }
 
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
         p.data -= 0.005 * p.grad;
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
         this.bias = new Value(Math.random());
         for(let i=0;i<this.n;i++){
             let arr = []
             for(let j=0;j<this.m;j++){
                 arr.push(new Value(Math.random()));
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
             sum = sum.operate('+',this.bias);
             res.push(sum);
         }
         // console.log('summmm',res[0].data,this.bias.data)
         return res;
     }
     parameters(){
         let ans = [];
         for(let i=0;i<this.n;i++){
             for(let j=0;j<this.m;j++){
                 ans.push(this.weights[i][j]);
             }
         }
         ans.push(this.bias);
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
     let ans = [];
     for(let o of input){
         ans.push(o.operate('relu',null));
     }
     return ans;
 }
 
 function sigmoid_f(input){
     let ans = [];
     for(let o of input){
         ans.push(o.operate('sigmoid'));
     }
     return ans;
 }
 
 
 function softmax(input){
     let ans = [];
     let sum = new Value(0);
     for(let o of input){
         let tmp = o.operate('exp');
         sum = sum.operate('+',tmp);
         ans.push(tmp);
     }
     for(let i=0;i<ans.length;i++){
         ans[i] = ans[i].operate('/',sum);
     }
     return ans;
 }
 
 
 class MLP{
     constructor(){
         this.linear1 = new Linear(2,2);
         this.linear2 = new Linear(2,2);
         this.linear3 = new Linear(2,1);
     }
     forward(x,y){
         // x B,T
         // y B
         let predicts = []; 
         for(let i=0;i<x.length;i++){
             let _x = x[i]
             _x = this.linear1.forward(_x);
             // console.log('zzz',_x[0].data);
             _x = relu(_x);
             // console.log('yyy',_x[0].data);
             _x = this.linear2.forward(_x);
             _x = relu(_x);
             _x = this.linear3.forward(_x);
             // let p = _x[0].operate('sigmoid',null);
             // console.log('xxxx',_x[0].data);
             predicts.push(_x[0].operate('sigmoid'));
         }
         if(y == null){
             return [null,predicts]
         }else{
             let loss = new Value(0);
             for(let i=0;i<y.length;i++){
                 let _y = y[i]
                 let p = predicts[i];
                 let epsilon = 1e-15
                 if(p < epsilon){
                     p = epsilon;
                 }
                 if(p > 1-epsilon){
                     p = 1 - epsilonl;
                 }
                 let _loss = 0
                 if(_y == 1){
                     _loss = p.operate('log').operate('*',_y).operate('*',-1);
                 }else{
                     _loss = p.operate('*',-1).operate('+',1).operate('log').operate('*',1-_y).operate('*',-1);
                 }
                 loss = loss.operate("+",_loss);
             }
             return [loss,predicts]
         }
     }
     parameters(){
         let ans1 = this.linear1.parameters();
         let ans2 = this.linear2.parameters();
         return ans1.concat(ans2);
     }
 }
 
 
 class MLPSoftmax{
     constructor(){
         this.linear1 = new Linear(2,3);
         this.linear2 = new Linear(3,2);
     }
     forward(x,y){
         // x B,T
         // y B
         let predicts = []; // B 2
         for(let i=0;i<x.length;i++){
             let _x = x[i]
             _x = this.linear1.forward(_x);
             // console.log('zzz',_x[0].data);
             _x = sigmoid_f(_x);
             // console.log('yyy',_x[0].data);
             _x = this.linear2.forward(_x);
             // let p = _x[0].operate('sigmoid',null);
             // console.log('xxxx',_x[0].data);
             _x = softmax(_x)
             predicts.push(_x)
         }
         if(y == null){
             return [null,predicts]
         }else{
             let loss = new Value(0);
             for(let i=0;i<y.length;i++){
                 let logit = predicts[i][y[i]].operate('log')
                 loss = loss.operate("-",logit);
             }
             return [loss,predicts]
         }
     }
     parameters(){
         let ans1 = this.linear1.parameters();
         let ans2 = this.linear2.parameters();
         return ans1.concat(ans2);
     }
 }
 
 
 
 function test6(){
     let mlp = new MLPSoftmax();
     let parameters = mlp.parameters();
     let x = [1,2];
     let y = 1;
     for(let i=0;i<100000;i++){
         let arr = mlp.forward([x],[y]);
         loss = arr[0]
         zero_grad(parameters);
         backward(loss);
         step_grad(parameters);
         console.log(loss.data);
     }
 }
 
 
 function train_model(x,y){
     let mlp = new MLPSoftmax();
     let parameters = mlp.parameters();
     // for(let i=0;i<x.length;i++){
     //     let _x = x[i];
     //     let _y = y[i];
     //     arr = mlp.forward([_x],null);
     //     console.log(_x,_y,arr[1][0][0].data);
     // }
     for(let j=0;j<20000;j++){
         arr = mlp.forward(x,y);
         loss = arr[0]
         zero_grad(parameters);
         backward(loss);
         step_grad(parameters);
         if(j % 1000 == 0){
             console.log(loss.data);
         }
     }
     for(let i=0;i<x.length;i++){
         let _x = x[i];
         let _y = y[i];
         arr = mlp.forward([_x],null);
         console.log(_x,_y,arr[1][0][_y].data);
     }
     return mlp;
 }

// let x = [
//     [0.,0.],
//     [1,0],
//     [1,1],
//     [0,1]
// ];
// let y = [
//     1,
//     0,
//     1,
//     0
// ];

// train_model(x,y)