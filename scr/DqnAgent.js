//import * as tf from '@tensorflow/tfjs-node-gpu'
import * as tf from '@tensorflow/tfjs-node'
import loadsh from 'loadsh'

const f_log = './log'
const f_mode = './models'

const weights_filename = 'dqn_model_weights.hdf5'

const simple_model_filename = 'dqn_model_simple.yaml'
const simple_weights_filename = 'dqn_model_weights_simple.hdf5'

const INITIAL_EXPLORATION = 1.0
const FINAL_EXPLORATION = 0.1
const EXPLORATION_STEPS = 500

//losses の定義を書く

let losses = { 
        'main_output': function(y_true,y_pred){
        if(y_pred){
            return tf.backend(y_pred)
        }else{
            return y_true
        }
    }
}

export async function loss_func(y_true,y_pred,a){
    let error = tf.abs(y_pred - y_true)
    let quadratic_part = tf.clipByValue(error, 0.0, 1.0)
    let linear_part = error - quadratic_part
    let loss = tf.sum(0.5 * tf.square(quadratic_part) + linear_part)
    //tf.scalar(loss)

    return loss
}

export async function customize_loss(args){
    y_true, y_pred, a = args
    a_one_hot = tf.oneHot(a,backend.shape(y_pred)[1],1.0,0.0)
    q_value = tf.reduce_sum(tf.mul(y_pred, a_one_hot), reduction_indices=1)
    error = tf.abs(q_value - y_true)
    quadratic_part = tf.clipByValue(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    
    return loss
}

export class DqnAgent{
    constructor(enable_actions,environment_name,graves,ddqn){
        this.name = "DQNAgent"
        this.environment_name = environment_name
        this.enable_actions = enable_actions
        this.n_actions = this.enable_actions.length
        this.minibatch_size = 32
        this.replay_memory_size = 5000
        this.learning_rate = 0.00025
        this.discount_factor = 0.9
        this.use_graves = graves
        this.use_ddqn = ddqn
        this.exploration = INITIAL_EXPLORATION
        this.exploration_step = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / EXPLORATION_STEPS
        this.model_dir = '/mnt/host/kerajs/src/models'
        this.model_name = `${this.environment_name}.ckpt`

        //this.old_session = tf.getBackend()
        //this.session = tf.Session('')
        //tf.backend.tensolflow_backedn.setSettion(this.session)

        this.D = Array()

        this.current_loss = 0.0
    }

    async init_model(){
        let state_input = tf.layers.input({shape:[1,16,16],name:'state'})
        let action_input = tf.layers.input({shape:[null],name:'action',dtype:'int32'})
        let x = tf.layers.conv2d({ InputShape: [16, 4, 4], activation: 'relu', kernelSize: [2, 2], filters: 1 }).apply(state_input)
        x = tf.layers.conv2d({InputShape:[32,2,2],activation:'relu',kernelSize:[1,1],filters:3}).apply(x)
        x = tf.layers.conv2d({InputShape:[32,2,2],activation:'relu',kernelSize:[1,1],filters:3}).apply(x)
        x = tf.layers.flatten().apply(x)
        //console.log('flatten')
        x = tf.layers.dense({InputShape:[128],activation:'relu',units:3}).apply(x)
        //console.log('layers')

        let y_pred = tf.layers.dense({InputShape:[this.n_actions],activation:'linear',name:'main_output',units:3}).apply(x)
        let y_true = tf.layers.input({shape:[3,],name:'y_ture'})

        let loss_out = await loss_func(y_true,y_pred,action_input)
        this.model = tf.model({inputs:[state_input,action_input,y_true],outputs:[y_pred],name:'name'})

        let optimizer = tf.train.rmsprop({learning_rate:this.learning_rate})
        this.model.compile({loss:losses,//
                            optimizer:optimizer,
                            metrics:['accuracy']})
        this.target_modle = this.model
        this.summary_op = this.model.summary()//
        //this.summary_writer = tf.summary.fileWriter()//

    }

    async update_exploration(num){
        //console.log('update_exploration')
        if(this.exploration > FINAL_EXPLORATION){
            this.exploration -= this.exploration_step * num
            if (this.exploration < FINAL_EXPLORATION){
                this.exploration = FINAL_EXPLORATION
            }
        }
    }

    async Q_values(states,isTarget){
        //console.log('target model flag',isTarget)
        let model = null
        if(isTarget == true){
            model = this.target_modle 
        }else{
            model = this.model
        }
        let tmp_states = tf.tensor([states])
        let tmp_action = tf.tensor([0])
        let tmp_y_true = tf.tensor([Array(this.n_actions).fill(0)])
        let res = model.predict([tmp_states,tmp_action,tmp_y_true])
        //console.log('Q_values',res.arraySync()[0])
        return res.arraySync()[0]
    }

    async update_target_model(){
        this.target_model = await loadsh.cloneDeep(this.model)//clone_model(this.model)//util.jsで実装
    }

    async select_action(states,epsilon){
        //console.log("check epsilon",Math.random(),epsilon)
        if(Math.random() <= epsilon){
            let Qmax = Math.floor(Math.random()*3)
            //console.log("select_action True",Qmax)
            return Qmax
        }else{
            let Qmax = 0
            let tmp_arg = await this.Q_values(states)
            for(let tmp_v = 0; tmp_v < tmp_arg.length;tmp_v++){
                if(tmp_arg[0][tmp_v] > Qmax){
                    Qmax = tmp_arg[0][tmp_v]
                }
            }
            //console.log("select_action False",Qmax)
            return Qmax
        }
    }

    async store_experience(states,action,reward,states_1,terminal){
        this.D.push([states,action,reward,states_1,terminal])
        //console.log(this.D.length,this.replay_memory_size)
        if(this.D.length > this.replay_memory_size){
            this.D = this.D.slice(1)
            return true
        }
        return false
    }

    async experience_replay(step,score){
        let state_minibatch = []
        let y_minibatch = []
        let action_minibatch = []

        let minibatch_size = Math.min(this.D.length,this.minibatch_size)
        let minibatch_indexes = Array(minibatch_size)
        for(let i = 0 ; i < minibatch_size ; i++){
            minibatch_indexes[i] =  Math.floor(Math.random()*this.D.length)
        }
        //console.log(minibatch_indexes)
        //console.log("print D",this.D)
        for (let j of minibatch_indexes){
            //console.log("j",j,minibatch_size)
            let state_j = this.D[j][0]
            let action_j = this.D[j][1]
            let reward_j = this.D[j][2]
            let state_j_1 = this.D[j][3]
            let terminal = this.D[j][4]
            //console.log('action_j',action_j)
            let action_j_index = await this.enable_actions.indexOf(action_j)

            let y_j = await this.Q_values(state_j)
            let v = 1

            //console.log('y_j1',y_j)
            if (terminal){
                y_j[action_j_index] = Number(reward_j)
            }else{
                if (this.use_ddqn === false){
                    //console.log('not use ddqn')
                    let tmp_arg = await this.Q_values(state_j_1,true)
                    //console.log(tmp_arg)
                    for(let tmp_v = 0; tmp_v < tmp_arg.length;tmp_v++){
                        if(tmp_arg[0][tmp_v] > v){
                            v = tmp_arg[0][tmp_v]
                        }
                    }
                    //v = Math.max(tmp_arg[0])
                }else{
                    v = this.Q_values(state_j_1,true)[action_j_index]
                }
                //console.log('v1',v,y_j,reward_j,action_j_index)
                y_j[action_j_index] = Number(reward_j) + Number(this.discount_factor * v)
                //console.log('v2',v,y_j,reward_j)
            }
            //console.log('y_j2',y_j)
            state_minibatch.push(state_j)
            y_minibatch.push(y_j)
            action_minibatch.push(action_j_index)
            //console.log(y_minibatch)
        }
        let validation_data = null
        //console.log('score',score)
        if (score != null){
            //console.log('fit true')
            validation_data = [[tf.tensor(state_minibatch),tf.tensor(action_minibatch)],
                                tf.tensor(y_minibatch),
                                [Array(this.minibatch_size).fill(0),y_minibatch]]
            this.model.fit(
                [tf.tensor(state_minibatch),tf.tensor(action_minibatch)],
                tf.tensor(y_minibatch),
                {batchSIze:this.minibatch_size,epochs:1,verbose:0,validationData:validation_data}
            )
        }


        if (this.model.validation_data && DQNAgent.hasattr("summary_op")){
            val_data = this.model.validation_data
            tensors = this.model.inputs
            feed_dict = {}
            for (let i = 0 ; i < tensors.length; i++){
                feed_dict[tensors[i]] = val_data[i]
            }
            result = this.session.run([this.summary_op],{feed_dict:feed_dict})
            summary_str = result[0]
        }
        //score = this.model.predict({"state":state_minibatch,"action":action_minibatch,"y_true":y_minibatch})
        //console.log(state_minibatch)
        //console.log(action_minibatch)
        //console.log('y_minibatch',y_minibatch)
        //y_minibatch = tf.tensor(y_minibatch[0])
        //state_minibatch = tf.tensor(state_minibatch)
        //action_minibatch = tf.tensor(action_minibatch)
        //y_minibatch = tf.tensor(y_minibatch)
        //let sum_loss = this.model.predict([state_minibatch, action_minibatch, y_minibatch]).arraySync()[0][0]
        //console.log('before predict')
        let sum_loss=0
        for (let i = 0 ; i < y_minibatch[0].length; i++){ 
            let tmp_state_minibatch = tf.tensor([state_minibatch[i]])
            let tmp_action_minibatch = tf.tensor([action_minibatch[i]])
            //console.log(y_minibatch[i],y_minibatch[i][0])
            let tmp_y_minibatch = tf.tensor([y_minibatch[i]])
            sum_loss += this.model.predict([tmp_state_minibatch, tmp_action_minibatch, tmp_y_minibatch]).arraySync()[0][0]
        }
        //console.log(score.arraySync())
        //this.current_loss = score.arraySync()[0][0]
        this.current_loss = sum_loss/y_minibatch.length
    }

    load_model(model_path,simple){
    }
    save_model(num,simple){
    }

    end_session(){
        tf.backend.tensolflow_backend.setSettion(this.old_session)
    }
}