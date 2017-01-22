require 'torch'
require 'dp'
require 'rnn'
require 'GGClassReward0402'
require 'RecurrentGG0316'
require 'image'
require 'optim'
require 'CTMnist'
require 'CTmnist'
require 'CTmnist100'
require 'ReinforceIdentical'
require 'ReinforceLinear'


-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf


version = 12

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Recurrent Model for Visual Attention')
cmd:text('Example:')
cmd:text('$> th rnn-visual-attention.lua > results.txt')
cmd:text('Options:')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum',0.9, 'momentum')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 16, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1200, 'maximum number of epochs to run')
cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ reinforce ]]--
cmd:option('--rewardScale', 1, "scale of positive reward (negative is 0)")
cmd:option('--unitPixels', 49, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
cmd:option('--locatorStd', 0.03, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

--[[ glimpse layer ]]--
cmd:option('--glimpseHiddenSize', 256, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchSize', 12, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 4, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 256, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')

--[[ recurrent layer ]]--
cmd:option('--rho', 4, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 256, 'number of hidden units used in Simple RNN.')
cmd:option('--dropout', true, 'apply dropout on hidden neurons')
cmd:option('--dropoutRate', 0.2, 'apply dropout on hidden neurons')
cmd:option('--th', 0.4, 'apply dropout on hidden neurons')

--[[ glance Network ]]--
cmd:option('--downSampleFactor', 4, 'how much downsampling for you want for Glance Network')


--[[ data ]]--
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | TranslatedMnist | etc')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--noTest', false, 'dont propagate through the test set')
cmd:option('--overwrite', false, 'overwrite checkpoint')

cmd:text()

torch.setnumthreads(4)

local opt = cmd:parse(arg or {})
if not opt.silent then
    table.print(opt)
end
--
-- torch.setdefaulttensortype('torch.DoubleTensor')
--[[data]]--
-- ds = torch.checkpoint(
--    paths.concat(dp.DATA_DIR, 'checkpoint/dp.CTmnist.t7'),
--    function() return dp['CTmnist']() end,
--    opt.overwrite
-- )

ds = torch.checkpoint(
   paths.concat(dp.DATA_DIR, 'checkpoint/dp.CTmnist100.t7'),
   function() return dp['CTmnist100']() end,
   opt.overwrite
)

--[[Saved experiment]]--
if opt.xpPath ~= '' then
    if opt.cuda then
        require 'optim'
        require 'cunn'
        cutorch.setDevice(opt.useDevice)
    end
    xp = torch.load(opt.xpPath)
    if opt.cuda then
        xp:cuda()
    else
        xp:float()
    end
    print"running"
    xp:run(ds)
    os.exit()
end

--[[Model]]--

-- -- Glance Network
-- glanceNetwork = nn.Sequential()
-- glanceNetwork:add(nn.Convert(ds:ioShapes(), 'bchw'))
-- glanceNetwork:add(nn.SpatialAveragePooling(opt.downSampleFactor,opt.downSampleFactor,opt.downSampleFactor,opt.downSampleFactor))
-- ------------------------------------------------------------
-- -- convolutional network
-- ------------------------------------------------------------
-- -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
-- glanceNetwork:add(nn.SpatialConvolution(1, 32, 5, 5))
-- glanceNetwork:add(nn[opt.transfer]())
-- glanceNetwork:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
-- glanceNetwork:add(nn.SpatialConvolution(32, 64, 5, 5))
-- glanceNetwork:add(nn[opt.transfer]())
-- glanceNetwork:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- -- stage 3 : standard 2-layer MLP:
-- glanceNetwork:add(nn.Collapse(3))
-- glanceNetwork:add(nn.Linear(64, 200))
-- glanceNetwork:add(nn[opt.transfer]())
-- glanceNetwork:add(nn.Linear(200, #ds:classes()))

-- Glimpse Network

-- glimpse network (rnn input layer)
locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2))
locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
locationSensor:add(nn[opt.transfer]())


glimpseSensor = nn.Sequential()
glimpseSensor:add(nn.DontCast(nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale):float(),true))
glimpseSensor:add(nn.Collapse(3))
-- if opt.dropout then
--   glimpseSensor:add(nn.Dropout(0.5))
-- end
glimpseSensor:add(nn.Linear(ds:imageSize('c')*(opt.glimpsePatchSize^2)*opt.glimpseDepth, opt.glimpseHiddenSize))
glimpseSensor:add(nn[opt.transfer]())

glimpse = nn.Sequential()
glimpse:add(nn.ConcatTable():add(locationSensor):add(glimpseSensor))
glimpse:add(nn.JoinTable(1,1))
-- if opt.dropout then
--   glimpse:add(nn.Dropout(opt.dropoutRate))
-- end
glimpse:add(nn.Linear(opt.glimpseHiddenSize+opt.locatorHiddenSize, opt.imageHiddenSize))
glimpse:add(nn[opt.transfer]())
-- if opt.dropout then
--   glimpse:add(nn.Dropout(opt.dropoutRate))
-- end
glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))

-- rnn recurrent layer
-- GRU = nn.GRU(opt.imageHiddenSize, opt.hiddenSize)
recurrent = nn.Linear(opt.hiddenSize, opt.hiddenSize)
-- recurrent neural network
-- rnn = nn.Sequential()
-- rnn:add(glimpse):add(GRU)
rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn[opt.transfer](), 99999)

imageSize = ds:imageSize('h')
assert(ds:imageSize('h') == ds:imageSize('w'))

-- actions (locator)
locator = nn.Sequential()
locator:add(nn.Linear(opt.hiddenSize, 2))
locator:add(nn.MulConstant(0.5,true))
locator:add(nn.HardTanh()) -- bounds mean between -1 and 1
locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
assert(locator:get(3).stochastic == opt.stochastic or locator:get(4).stochastic == opt.stochastic, "Please update the dpnn package : luarocks install dpnn")
locator:add(nn.HardTanh()) -- bounds sample between -1 and 1
locator:add(nn.MulConstant(opt.unitPixels*2/ds:imageSize("h")))

attention = nn.RecurrentGG0316(rnn, locator, opt.rho, {opt.hiddenSize}, glanceNetwork,opt.unitPixels)


-- model is a reinforcement learning agent
agent = nn.Sequential()
agent:add(nn.Convert(ds:ioShapes(), 'bchw'))
agent:add(attention)


-- classifier :
agent:add(nn.SelectTable(-1))
if opt.dropout then
  agent:add(nn.Dropout(opt.dropoutRate))
end
agent:add(nn.ReinforceLinear(opt.hiddenSize, 200))
agent:add(nn[opt.transfer]())
if opt.dropout then
  agent:add(nn.Dropout(opt.dropoutRate))
end
agent:add(nn.ReinforceLinear(200, #ds:classes()))
agent:add(nn.LogSoftMax())

bl = nn.Sequential()
bl:add(nn.Constant(1,1)):add(nn.Add(1))

concat_bl = nn.ConcatTable():add(nn.Identity()):add(bl)
concat_classpred = nn.ConcatTable():add(nn.Identity()):add(concat_bl)

agent:add(concat_classpred)

--concat classpred, classpred, baseline and glanceNetwork output
-- concat_agent_glanceNetwork = nn.ConcatTable():add(agent):add(glanceNetwork)

network = nn.Sequential()
network:add(agent)
-- network:add(concat_agent_glanceNetwork)
--
-- network:add(nn.FlattenTable()) --here the network outputs a flatten table
-- get_subtab = nn.ConcatTable():add(nn.SelectTable(2)):add(nn.SelectTable(3))
--
-- rearrange = nn.ConcatTable():add(nn.SelectTable(1)):add(get_subtab):add(nn.SelectTable(4))
--
-- network:add(rearrange)


-- output will be sth like
-- {
--   1 :
--     {
--       1 : DoubleTensor - size: 16x10
--       2 :
--         {
--           1 : DoubleTensor - size: 16x10
--           2 : DoubleTensor - size: 16x1
--         }
--     }
--   2 : DoubleTensor - size: 16x10
-- }

-- print(network:forward(torch.rand(20,28,28,1)))
--initialization of param
if opt.uniform > 0 then
    for k,param in ipairs(network:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
    end
end


VRClassReward = nn.GGClassReward0402(network, opt.rewardScale, opt.glimpsePatchSize, glanceNetwork, attention,ds:imageSize('h'),opt.th)

--[[Propagators]]--
opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch

train = dp.Optimizer{
    loss = nn.ParallelCriterion(true) --weighted sum of different criterion; present {classpred, {classpred, basereward}} to two criterion, share the same target
        :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
        :add(nn.ModuleCriterion(VRClassReward, nil, nn.Convert())) -- REINFORCE
        -- :add(nn.ModuleCriterion(nn.CrossEntropyCriterion(), nil, nn.Convert())) -- BACKPROP for glanceNetwork
    ,
    epoch_callback = function(model, report) -- called every epoch
        if report.epoch > 0 then
            opt.learningRate = opt.learningRate + opt.decayFactor
            opt.learningRate = math.max(opt.minLR, opt.learningRate)
            if not opt.silent then
              print("learningRate", opt.learningRate)
              print('baseline',bl.output:mean(1))
              print('vr',VRClassReward.vrReward:mean(1))
              --print('actions',attention.actions[1][1],attention.actions[2][1],attention.actions[3][1],attention.actions[4][1])
            end
    end
    end,

    callback = function(model, report)

        if opt.cutoffNorm > 0 then
            local norm = model:gradParamClip(opt.cutoffNorm) -- affects gradParams
            opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
            --if opt.lastEpoch < report.epoch and not opt.silent then
                --print("mean gradParam norm", opt.meanNorm)
            --end
        end
        model:updateGradParameters(opt.momentum) -- affects gradParams
        model:updateParameters(opt.learningRate) -- affects params
        model:maxParamNorm(opt.maxOutNorm) -- affects params
        model:zeroGradParameters() -- affects gradParams
    end,
    feedback = dp.Confusion{output_module=nn.SelectTable(1)},
    sampler = dp.ShuffleSampler{
        epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
    },
    progress = opt.progress
}


valid = dp.Evaluator{
    feedback = dp.Confusion{output_module=nn.SelectTable(1)},
    sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
    progress = opt.progress
}


if not opt.noTest then
    tester = dp.Evaluator{
        feedback = dp.Confusion{output_module=nn.SelectTable(1)},
        sampler = dp.Sampler{batch_size = opt.batchSize}
    }
end

--[[Experiment]]--
xp = dp.Experiment{
    model = network,
    optimizer = train,
    validator = valid,
    tester = tester,
    observer = {
        ad,
        dp.FileLogger(),
        dp.EarlyStopper{
            max_epochs = opt.maxTries,
            error_report={'validator','feedback','confusion','accuracy'},
            maximize = true
        }
    },
    random_seed = os.time(),
    max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.useDevice)
    xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
    -- print"Network :"
    -- print(network)
end

xp.opt = opt

xp:run(ds)
