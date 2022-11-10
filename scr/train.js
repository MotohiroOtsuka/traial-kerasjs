import {CatchBall} from './CatchBall.js'
import {DqnAgent} from './DqnAgent.js'

//parameters
const nEpocs = 1000

//enviroment,agent
const env = new CatchBall(true,false)
const agent = new DqnAgent(env.enable_actions,env.name,false,false)
await agent.init_model()

//variables
let win = 0
let totalFrame = 0
let e = 0

let doReplayCount = 0

while (e < nEpocs){
    //console.log("start next epoch")
    let frame = 0
    let loss = 0.0
    let Qmax = 0.0
    //console.log("reset env")
    await env.reset()
    //console.log("set observe")
    let tmp = await env.observe()
    let state_t_1 = tmp[0] 
    let reward_t = tmp[1]
    let tarminal = tmp[2]
    let start_replay

    win = 0
    //console.log(state_t_1, reward_t,tarminal)
    //console.log("epoch start")
    while (tarminal == false){
        let statet_t = state_t_1
        
        //excute action in environment
        //console.log("select_action")
        let action_t = await agent.select_action([statet_t],agent.exploration)
        //console.log(action_t)
        await env.execute_action(action_t)
        tmp = await env.observe()
        state_t_1 = tmp[0] 
        reward_t = tmp[1]
        tarminal = tmp[2]

        start_replay = await agent.store_experience([statet_t],action_t,reward_t,[state_t_1],tarminal)
        //console.log('1',start_replay,tarminal)
        if(start_replay){
            //console.log('start_replay True')
            doReplayCount += 1
            await agent.update_exploration(e)
            if (doReplayCount > 2){
                //console.log('start_replay > 2 True')
                await agent.experience_replay(e)
                doReplayCount = 0 
            }
        }

        if(totalFrame % 500 == 0 && start_replay){
            //console.log('update target model')
            await agent.update_target_model()
        }
        //console.log('next frame')
        frame += 1
        totalFrame += 1
        loss += agent.current_loss
        //Qmax = Math.max(agent.Q_values([statet_t]))
        let tmp_arg = await agent.Q_values([statet_t])
        for(let tmp_v = 0; tmp_v < tmp_arg.length;tmp_v++){
            if(tmp_arg[0][tmp_v] > Qmax){
                Qmax = tmp_arg[0][tmp_v]
            }
        }
        //console.log(reward_t)
        if (reward_t == 1){
            win += 1
        }
        //console.log('2',start_replay,tarminal)
    }
    //console.log('3',start_replay,tarminal,Qmax,frame)
    if(start_replay==true){
        //console.log('start_repay_flag true')
        await agent.experience_replay(e,win)
        console.log(`epoch: ${e+1} | WIN: ${win} | LOSS: ${frame} | Q_max: ${Qmax/frame}`)
        win = 0
    }
    
    //saveするコードを追加する
    if(start_replay){
        e += 1
    }    
    //console.log("epoch",e)
}
//saveするコードを追加する
console.log("end all epoch")