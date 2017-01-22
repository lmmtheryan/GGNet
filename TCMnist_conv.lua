require 'torch'
require 'dp'
require 'rnn'
require 'GGClassReward'
require 'RecurrentGG'
require 'image'
require 'CTmnist'

-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf


version = 12

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Glance Network on TCMNIST')
cmd:text('Example:')
cmd:text('Options:')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 800, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of contatenation of all gradParam tensors')
cmd:option('--batchSize', 16, 'number of examples per batch')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 400, 'maximum number of epochs to run')
cmd:option('--maxTries', 100, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--transfer', 'ReLU', 'activation function')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')

--[[ reinforce ]]--
cmd:option('--rewardScale', 1, "scale of positive reward (negative is 0)")
cmd:option('--unitPixels', 13, "the locator unit (1,1) maps to pixels (13,13), or (-1,-1) maps to (-13,-13)")
cmd:option('--locatorStd', 0.11, 'stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')
cmd:option('--stochastic', false, 'Reinforce modules forward inputs stochastically during evaluation')

--[[ glimpse layer ]]--
cmd:option('--glimpseHiddenSize', 128, 'size of glimpse hidden layer')
cmd:option('--glimpsePatchSize', 8, 'size of glimpse patch at highest res (height = width)')
cmd:option('--glimpseScale', 2, 'scale of successive patches w.r.t. original input image')
cmd:option('--glimpseDepth', 1, 'number of concatenated downscaled patches')
cmd:option('--locatorHiddenSize', 128, 'size of locator hidden layer')
cmd:option('--imageHiddenSize', 256, 'size of hidden layer combining glimpse and locator hiddens')

--[[ recurrent layer ]]--
cmd:option('--rho', 7, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 256, 'number of hidden units used in Simple RNN.')
cmd:option('--dropout', true, 'apply dropout on hidden neurons')

cmd:option('--downSampleFactor', 2, 'how much downsampling for you want for Glance Network')

--[[ data ]]--
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | TranslatedMnist | etc')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--validEpochSize', -1, 'number of valid examples used for early stopping and cross-validation')
cmd:option('--noTest', false, 'dont propagate through the test set')
cmd:option('--overwrite', false, 'overwrite checkpoint')
cmd:option('--accUpdate', false, 'accUpdateGradParameters')

cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
    table.print(opt)
end

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

-- torch.setdefaulttensortype('torch.DoubleTensor')

--[[data]]--
-- ds = torch.checkpoint(
--    paths.concat(dp.DATA_DIR, 'checkpoint/dp.CTmnist.t7'),
--    function() return dp['CTmnist']() end,
--    opt.overwrite
-- )


ds = torch.checkpoint(
   paths.concat(dp.DATA_DIR, 'checkpoint/dp.CTmnist.t7'),
   function() return dp['CTmnist']() end,
   opt.overwrite
)

torch.setnumthreads(4)
-- ds=dp['Mnist']()

-- input = ds:get()
-- testnn = nn.Sequential()
-- testnn:add(nn.SpatialAveragePooling(2,2,2,2))
-- output=testnn:forward(input[1]:reshape(1,60,60):double())
-- image.save('test_in.png', input[1]:reshape(60,60))
-- image.save('test_out.png', output)



--[[Model]]--
network = nn.Sequential()
network:add(nn.Convert(ds:ioShapes(), 'bchw'))
network:add(nn.SpatialAveragePooling(opt.downSampleFactor,opt.downSampleFactor,opt.downSampleFactor,opt.downSampleFactor))
------------------------------------------------------------
-- convolutional network
------------------------------------------------------------
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
network:add(nn.SpatialConvolution(1, 32, 5, 5))
-- network:add(nn.SpatialBatchNormalization(32))
network:add(nn.ReLU())
network:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
network:add(nn.SpatialConvolution(32, 64, 5, 5))
-- network:add(nn.SpatialBatchNormalization(64))
network:add(nn.ReLU())
network:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 2-layer MLP:
network:add(nn.Collapse(3))
network:add(nn.Linear(64*2*2, 200))
-- network:add(nn.BatchNormalization(200))
network:add(nn.ReLU())
network:add(nn.Linear(200, #ds:classes()))



--nitialization of param
if opt.uniform > 0 then
    for k,param in ipairs(network:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
    end
end

--[[Propagators]]--
opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch

train = dp.Optimizer{
    loss = nn.ModuleCriterion(nn.CrossEntropyCriterion(), nil, nn.Convert()), -- BACKPROP
    epoch_callback = function(model, report) -- called every epoch
        if report.epoch > 0 then
            opt.learningRate = opt.learningRate + opt.decayFactor
            opt.learningRate = math.max(opt.minLR, opt.learningRate)
            if not opt.silent then
                print("learningRate", opt.learningRate)
            end
        local filename ='mnistConv_Tanh.net'
        if paths.filep(filename) then
           os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
        end
        print('<trainer> saving network to '..filename)
        torch.save(filename, model)
    end
    end,

    callback = function(model, report)
        if opt.accUpdate then
            model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
        else
            model:updateGradParameters(opt.momentum) -- affects gradParams
            model:updateParameters(opt.learningRate) -- affects params
        end
        model:maxParamNorm(opt.maxOutNorm) -- affects params
        model:zeroGradParameters() -- affects gradParams
    end,
    feedback = dp.Confusion(),
    sampler = dp.ShuffleSampler{
        epoch_size = opt.trainEpochSize, batch_size = opt.batchSize
    },
    progress = opt.progress
}


valid = dp.Evaluator{
    feedback = dp.Confusion(),
    sampler = dp.Sampler{epoch_size = opt.validEpochSize, batch_size = opt.batchSize},
    progress = opt.progress
}
if not opt.noTest then
    tester = dp.Evaluator{
        feedback = dp.Confusion(),
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
    print"Network :"
    print(network)
end

xp.opt = opt

xp:run(ds)
