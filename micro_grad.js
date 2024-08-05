class Value{
    constructor(data){
        this.data = data;
    }
    add(data){
        if(data instanceof Value){
            return Value(this.data + this.data.data);
        }else{
            return Value(this.data + data);
        }
    }
    sub(data){
        if(data instanceof Value){
            return Value(this.data - this.data.data);
        }else{
            return Value(this.data - data);
        }
    }
    multi(data){
        if(data instanceof Value){
            return Value(this.data * this.data.data);
        }else{
            return Value(this.data * data);
        }
    }
    div(data){
        if(data instanceof Value){
            return Value(this.data + this.data.data);
        }else{
            return Value(this.data + data);
        }
    }
}