class Module{

    constructor(){
        this._module = {};
        this._parameters = {};
        this._non_parameters = {}
        this.training = true
    }

    modules(){
        return Object.values(this._module);
    }

    train(){
        this.training = true
        for(let key in this._module){
            this._module[key].train();
        }
    }

    eval(){
        this.training = false
        for(let key in this._module){
            this._module[key].eval();
        }
    }

    named_parameters(){
        let ans = [];
        for(let key in this._parameters){
            ans.push([key,this._parameters[key]]);
        }
        for(let key in this._module){
            let params = this._module[key].named_parameters();
            for(let o of params){
                ans.push([key+'.'+o[0],o[1]])
            }
        }
        return ans;
    }

    parameters(){
        let ans = Object.values(thia._parameters);
        for(let key of this._module){
            ans.concat(this._module[key].parameters());
        }
        return ans;
    }

    add_parameter(k,v){
        val = Parameter(v,k)
        this._parameters[k] = val
        return val;
    }

    set(key,val){
        if (val instanceof Parameter){
            this._parameters[key] = val;
        }else if(val instanceof Module){
            this._module[key] = val;
        }else{
            this._non_parameters[key] = val;
        }
    }

    call(...args) {
        return this.forward(...args);
    }

    get(key){
        if(key in this._parameters){
            return this._parameters[key]
        }
        if(key in this._module){
            return this._module[key]
        }
        return null;
    }

    forward(){

    }

    str(){

    }
}



class Parameter{
    constructor(x,name){
        this.value = x;
        this.name = name;
    }
    update(x){
        this.value = x;
    }
    str(){

    }
}


class ModuleA1 extends Module{
    constructor(){
        super()
        this.set('p1',new Parameter(5));
        this.set('non_param' , 10);
        this.set('a',new ModuleA2());
        this.set('b' , new ModuleA3());
    }
}

class ModuleA2 extends Module{
    constructor(){
        super()
        this.set('p2',new Parameter(10))
    }
}

class ModuleA3 extends Module{
    constructor(){
        super()
        this.set('c',new ModuleA4())
    }
}

class ModuleA4 extends Module{
    constructor(){
        super()
        this.set('p3',new Parameter(15))
    }
}


// mod = new ModuleA1()
// console.log(mod.named_parameters())


function mul(x,y){
    return x*y;
}

function id(x){
    return x;
}

function  add(x,y){
    return x + y;
}

function neg(x){
    return -x;
}

function lt(x,y){
    if(x < y){
        return 1.0;
    }
    return 0.0;
}

function eq(x,y){
    if(x == y){
        return 1.0;
    }
    return 0.0;
}


function max(x,y){
    if(x > y){
        return x;
    }
    return y;
}


function is_close(x,y){
    if(x < y){
        a = x;
        x = y;
        y = a;
    }
    if(x - y < 0.02){
        return 1.0;
    }
    return 0.0;
}


function sigmoid(x){
    if(x > 0){
        return 1.0 / (1 + exp(-x));
    }else{
        return exp(x) / (1 + exp(x));
    }
}


function relu(x){
    return max(x,0);
}

EPS  = 1e-6



function log(x){
    return Math.log(x + EPS);
}

function exp(x){
    return Math.exp(x);
}

function log_back(x,d){
    return d / (x+EPS);
}

function inv(x){
    return 1 / x;
}

function inv_back(x,d){
    return -d / (x*x + EPS)
}

function relu_back(x,d){
    if(x > 0){
        return d;
    }
    return 0
}

function map(fn){

    function _f(alist){
        let ans = []
        for(let o of alist){
            ans.push(fn(o));
        }
        return ans;
    }
    return _f
}


function neglist(ls){
    return map(neg)(ls)
}


function zipWith(fn){
    function _f(ls1,ls2){
        let ans = []
        for(let i=0;i<ls1.length;i++){
            ans.push(fn(ls1[i],ls2[i]));
        }
        return ans;
    }
    return _f
}

function addList(ls1,ls2){
    return zipWith(add)(ls1,ls2);
}


function reduce(fn,start){

    function _f(alist){
        let xx = start
        for(let i=0;i<alist.length;i++){
            xx = fn(alist[i],xx)
        }
        return xx
    }
    return _f
}


function sum(ls){
    return reduce(add,0)(ls)
}

function prod(ls){
    return reduce(mul,1)(ls)
}



class Context{
    constructor(no_grad){
        this.no_grad = no_grad
        this.saved_values = []
    }
    save_for_backward(values){
        if(this.no_grad){
            return
        }
        this.saved_values = values
    }

    saved_tensors(){
        return this.saved_tensors
    }
}

// operator instances
class ScalarFunction{

    _backward(ctx,d_out){
        return this.backward(ctx,d_out);
    }

    _forward(ctx,inps){
        return this.forward(ctx,inps);
    }

    apply(vals){
        let raw_vals = []
        let scalars = []
        for(let v of vals){
            if(v instanceof Scalar){
                scalars.push(v);
                raw_vals.push(v.data);
            }else{
                scalars.push(Scalar(v,[],null));
                raw_vals.push(v);
            }
        }
        let ctx = Context(false);
        c = this._forward(ctx,raw_vals);
        back = ScalarHistory(this,ctx,scalars);
        return Scalar(c,back)
    }
}

// scalar implemented
class Add extends ScalarFunction{

    forward(ctx,a,b){
        return a + b;
    }

    backward(ctx,d_out){
        return d_out,d_out
    }
}


class Log extends ScalarFunction{

    forward(ctx,a){
        ctx.save_for_backward(a)
        return log(a)
    }

    backward(ctx,d_out){
        [a,] = ctx.saved_values
        return log_back(a)
    }
}


class Mul extends ScalarFunction{

    forward(ctx,a,b){
        ctx.save_for_backward(a,b)
        return mul(a,b)
    }

    backward(ctx,d_out){
        [a,b] = ctx.saved_values
        return [d_out * b, d_out * a]
    }

}


class Inv extends ScalarFunction{

    forward(ctx,a){
        ctx.save_for_backward(a)
        return inv(a)
    }

    backward(ctx,d_out){
        [a,] = ctx.saved_values
        return d_out * inv_back(a)
    }
}


class Neg extends ScalarFunction{

    forward(ctx,a){
        ctx.save_for_backward(a)
        return neg(a)
    }

    backward(ctx,d_out){
        [a,] = ctx.saved_values
        return -d_out
    }
}



class Sigmoid extends ScalarFunction{

    forward(ctx,a){
        ctx.save_for_backward(a)
        return sigmoid(a)
    }

    backward(ctx,d_out){
        [a,] = ctx.saved_values
        return d_out * (1-sigmoid(a)) * sigmoid(a)
    }
}



class Relu extends ScalarFunction{

    forward(ctx,a){
        ctx.save_for_backward(a)
        return relu(a)
    }

    backward(ctx,d_out){
        [a,] = ctx.saved_values
        return relu_back(a,d_out) 
    }
}



class Exp extends ScalarFunction{

    forward(ctx,a){
        ctx.save_for_backward(a)
        return exp(a)
    }

    backward(ctx,d_out){
        [a,] = ctx.saved_values
        return exp(a) * d_out
    }
}


class LT extends ScalarFunction{

    forward(ctx,a,b){
        return lt(a,b)
    }

    backward(ctx,d_out){
        return [0,0]
    }
}


class EQ extends ScalarFunction{

    forward(ctx,a,b){
        return eq(a,b)
    }

    backward(ctx,d_out){
        return [0,0]
    }
}





class ScalarHistory{

    constructor(last_fn,ctx,inputs){
        this.last_fn = last_fn
        this.ctx = ctx
        this.inputs = inputs
    }
}

_var_count = 0

class Scalar{
    constructor(v,back,name){
        this.history = back
        this.derivative = null
        this.data = v
        this.unique_id = 0.9
        if(name != null){
            this.name = name
        }else{
            this.name = self.unique_id + ''
        }
    }
    str(){
        return "Scalar("+this.data+")"
    }

    mul(b){
        return Mul.apply(this,b)
    }

    div(b){
        return Inv.apply(this,b)
    }

    add(b){
        return Add.apply(this,b)
    }

    bool(){
        if(this.data){
            return true
        }
        return false
    }

    lt(b){
        return LT.apply(this,b)
    }

    gt(b){
        return LT.apply(b,this)
    }

    eq(b){
        return EQ.apply(this,b)
    }

    sub(b){
        return Add.apply(this,Mul.apply(-1,b))
    }

    neg(){
        return Mul.apply(this,-1)
    }

    log(){
        return Log.apply(this)
    }

    exp(){
        return Exp.apply(this)
    }

    sigmoid(){
        return Sigmoid.apply(this)
    }

    relu(){
        return Relu.apply(this)
    }

    accumulate_derivative(d){
        if(this.derivative == null){
            this.derivative = 0
        }
        this.derivative = this.derivative + d
    }

    is_leaf(){
        if(this.history && this.history.last_fn != null){
            return True
        }
        return false
    }

    is_constant(){
        return  this.history == null
    }

    parents(){
        return this.history.inputs
    }

    chain_rule(d_out){
        h = this.history
        derives = h.last_fn.backward(h.ctx,d_out)
        let ans = []
        for(let i=0;i<this.parents.length;i++){
            ans.push([this.parents[i],derives[i]]);
        }
        return ans
    }


    backward(d_out){
        if(d_out == null){
            d_out = 1.0
        }
        backprogate(this,d_out)
    }
}


function backprogate(variable,d_out){
    let ans = []
    function _f(node){
        for(let o of node.parents()){
            _f(o);
        }
        ans.push(node);
    }
    _f(variable);
    ans.reverse();
    variable.derivative = d_out
    cache = {}
    for(let o of ans){
        if(o.unique_id in caches){
            d = caches[o.unique_id]
        }else{
            d = o.derivative
        }
        if(!o.is_leaf()){
            let tmp = o.chain_rule(d)
            for(let [p,d1] of tmp){
                if(p.is_leaf()){
                    p.accumulate_derivative(d1)
                }else{
                    if(!(p.unique_id in caches)){
                        caches[p.unique_id] = 0;
                    }
                    caches[p.unique_id] = caches[p.unique_id] + d1
                }
            }
        }
    }
}
