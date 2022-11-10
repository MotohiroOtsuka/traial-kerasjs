

export class Ball{
    constructor(col){
        this.col = col
        this.row = 0
    }

    update(){
        this.row += 1
    }

    isDroped(n_rows){
        if(this.row >= n_rows){
            return true
        }
        return false
    }
}

export class CatchBall{
    constructor(time_limit,simple){
        this.name = "train_name"
        this.screen_n_rows = 16
        this.screen_n_cols = 16
        this.player_length = 3
        this.enable_actions = [0,1,2]
        this.frame_rate = 5
        this.ball_post_interbal = 4
        this.ball_past_time = 0 
        this.past_time = 0
        this.balls = []
        this.time_limit = time_limit
        this.simple = simple        

        this.reset()
    }

    async update(action){
        /*action
            0: do nothing
            1: move left
            2: move right
        */
        //console.log('update1 player_col',this.player_col)
        if (action == this.enable_actions[1]){//move left
            if(this.player_col - 1 > 0){
                this.player_col -= 1 
            }else{
                this.player_col = 1 
            }
        }else if(action == this.enable_actions[2]){//move right
            if(this.player_col + 1 < this.screen_n_cols - this.player_length){
                this.player_col = this.player_col + 1
            }else{
                this.player_col = this.screen_n_cols - this.player_length
            }
        }
        //console.log('update2 player_col',this.player_col)
        //console.log(this.balls)
        for (let i=0 ; i < this.balls.length; i++ ){
            this.balls[i].row += 1
        }
        //console.log(this.ball_past_time,this.ball_post_interbal)
        if (this.ball_past_time == this.ball_post_interbal) {
            this.ball_past_time = 0
            let new_pos = Math.floor(Math.random()*this.screen_n_cols)
            if(!this.simple){
                while(
                    this.balls.length > 0 &&
                    (
                        (Math.abs(new_pos - this.balls[this.balls.length-1].col) > this.ball_post_interbal + this.player_length - 1 ) ||
                        (Math.abs(new_pos - this.balls[this.balls.length-1].col) < this.player_lenght)
                    )
                ){
                    new_pos = Math.floor(Math.random()*this.screen_n_cols)
                }
            }
            this.balls.push(new Ball(new_pos))
        }else{
            this.ball_past_time += 1
        }  
        //collision detection
        this.reward = 0
        this.terminal = false
        //console.log(this.time_limit,this.past_time)
        this.past_time += 1
        if(this.time_limit && this.past_time > 500){
            this.terminal = true
        }
        //console.log(this.balls)
        if(this.balls[0].row == this.screen_n_rows -1){
            //console.log(this.player_col,this.balls[0].col,this.player_col + this.player_length)
            if((this.player_col <= this.balls[0].col) && (this.balls[0].col < this.player_col + this.player_length)){
                //catch
                //console.log('catch')
                this.reward = 1
            }else{
                //drop
                //console.log('drop')
                this.reward = -1
                this.terminal = true
            }
        }
        //console.log(this.balls)
        let new_balls = []
        for (let i=0 ; i < this.balls.length; i++ ){       
            if (!this.balls[i].isDroped(this.screen_n_rows)){
                new_balls.push(this.balls[i])
            }
        }
        this.balls = new_balls
    }

    async draw(){
        this.screen = new Array(this.screen_n_rows)
        for (let i = 0 ; i < this.screen_n_rows ; i ++){
            this.screen[i] = new Array(this.screen_n_clos)
            for(let j = 0 ; j < this.screen_n_cols; j++ ){
                this.screen[i][j] = 0
            }
        }
        for (let i = this.player_col; i< this.player_length ; i++){
            this.screen[this.player_row][i] = 1;
        }
        for (let i = 0 ; i < this.balls.length ; i++){
            this.screen[this.balls[i].row][this.balls[i].cal]=0.5
        }
    }

    async observe(){
        await this.draw()
        
        //console.log(this.screen)
        return [this.screen,this.reward,this.terminal]
    }

    async execute_action(action){
        await this.update(action)
    }

    async reset(){
        //reset plyaer position
        this.player_row = this.screen_n_rows - 1
        //console.log(this.screen_n_cols,this.player_length)
        this.player_col = Math.floor(Math.random()*(this.screen_n_cols - this.player_length))//player col の値がどこかでNaNになっている。
        //console.log("reset player_col",this.player_col)
        //reset ball position
        this.balls = []
        this.balls.push(new Ball(Math.floor(Math.random()*this.screen_n_cols)))

        //reset other variables
        this.reward = 0
        this.terminal = false
        this.past_time = 0
        this.ball_past_time = 0
    }
}