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


