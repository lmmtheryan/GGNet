------------------------------------------------------------------------
--[[ VRClassReward ]]--
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRClassReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local GGClassReward0402, parent = torch.class("nn.GGClassReward0402", "nn.Criterion")

function GGClassReward0402:__init(module, scale, glipmseSize, glance, attention, imageHeight,th)
    parent.__init(self)
    self.module = module -- so it can call module:reinforce(reward)
    self.scale = scale or 1 -- scale of reward
    self.criterion = nn.MSECriterion() -- baseline criterion
    self.sizeAverage = true
    self.glimpseSize = glipmseSize
    self.gradInput = {torch.Tensor()}
    --self.glanceModel = glance:clone('weight','bias')
    if imageHeight == 100 then
      self.glanceModel = torch.load('mnistConv100_Relu.net').module
    else
      self.glanceModel = torch.load('mnistConv60_Relu.net').module
    end
    self.th = th or 0.4
    self.attention = attention
end

function GGClassReward0402:updateOutput(input, target)
    --inputs are {classpred,actions,sample batch}
    assert(torch.type(input) == 'table')

    local classpred = self:toBatch(input[1], 1)
    local actions = self.attention.actions
    local optCuda = torch.type(classpred) == 'torch.CudaTensor'
    local input_ = optCuda and self.attention.inputImages:cuda() or self.attention.inputImages

    self._maxVal = self._maxVal or classpred.new()
    self._maxIdx = self._maxIdx or torch.type(classpred) == 'torch.CudaTensor' and classpred.new() or torch.LongTensor()

    -- max class value is class prediction
    self._maxVal:max(self._maxIdx, classpred, 2) --store the maxIdx in self._maxIdx; return the maxVal to self._maxVal
    if torch.type(self._maxIdx) ~= torch.type(target) then
        self._target = self._target or self._maxIdx.new()
        self._target:resize(target:size()):copy(target)
        target = self._target
    end

    -- reward = scale when correctly classified
    self._reward = self._maxIdx.new()
    self._reward:eq(self._maxIdx, target)
    self.reward = self.reward or classpred.new()
    self.reward:resize(self._reward:size(1)):copy(self._reward)
    --need to modify the following
    --local posterior = nn.SoftMax()
    --local post = torch.zeros(input_:size(1))
    local prediction = optCuda and torch.zeros(input_:size(1)):cuda() or torch.zeros(input_:size(1))
    local tempGradInput = self.glanceModel.gradInput
    local tempOutput = self.glanceModel.output
    local w = optCuda and torch.zeros(input_:size()):cuda() or torch.zeros(input_:size())
    for i = 1,input_:size(1) do
          local cs =  self.glanceModel:forward(input_[i])
          -- local err  = salCriterion:forward(cs,target[i])
          -- local dedo = salCriterion:backward(cs,target[i])
          --local temp =  posterior:forward(cs)
          --post[i] = temp[target[i]]
          local dedo = optCuda and torch.zeros(classpred:size(2)):cuda() or torch.zeros(classpred:size(2))
          dedo[target[i]] = -1
          w[i] = self.glanceModel:backward(input_[i],dedo)-- dSc/dI
          --w[i] = (-w[i]):cmax(0)
          w[i]:abs()
          w[i] = w[i]:div(w[i]:max())
          w[i] = w[i]:cmax(self.th):add(-self.th)
          -- w[i] = w[i]:cmul(post):add(1-post)
          --0prediction[i] =math.exp(classpred[i][target[i]])
        end
    -- print(self.glanceModel.gradInput)
    self.glanceModel.gradInput = tempGradInput
    self.glanceModel.output = tempOutput
    sumW = w:sum(2):sum(3):sum(4):resizeAs(self.reward)
    local nstep = table.getn(actions)
    local inputMask = optCuda and torch.zeros(input_:size(1),input_:size(3),input_:size(4)):cuda() or torch.zeros(input_:size(1),input_:size(3),input_:size(4))
    for i=1,nstep do
        actions[i]:add(1)
        actions[i]:div(2)
        actions[i][{{},1}]:mul(input_:size(3))
        actions[i][{{},2}]:mul(input_:size(4))
        actions[i]:ceil()

        for j=1,input_:size(1) do
            inputMask[{j,{},{}}]:sub(actions[i][{j,1}],math.min(actions[i][{j,1}]+self.glimpseSize,input_:size(3)),actions[i][{j,2}],math.min(actions[i][{j,2}]+self.glimpseSize,input_:size(4))):fill(1)
        end
    end

    --rw:cmul(post):add(-post):add(1)

    --self.reward:cmul(rw):cmul(prediction)
    self.rw = self.reward.new()
    self.rw = inputMask:cmul(w)
    self.rw = self.rw:sum(2):sum(3):resizeAs(self.reward):cdiv(sumW):mul(self.scale)
    -- loss = -sum(reward)
    self.reward:cmul(self.rw)
    self.output = -self.reward:sum()
    if self.sizeAverage then
        self.output = self.output/classpred:size(1)
    end
    return self.output
end



function GGClassReward0402:updateGradInput(inputTable, target)
    local input = self:toBatch(inputTable[1], 1)
    local baseline = self:toBatch(inputTable[2], 1)

    -- reduce variance of reward using baseline
    self.vrReward = self.vrReward or self.reward.new()
    self.vrReward:resizeAs(self.reward):copy(self.reward)
    self.vrReward:add(-1, baseline)
    if self.sizeAverage then
      self.vrReward:div(input:size(1))
    end
    -- broadcast reward to modules
    self.module:reinforce(self.vrReward)
    local m = self.rw
    -- self.module:reinforceNVR(m:div(m:mean()))
    self.module:reinforceNVR(m)
    -- zero gradInput (this criterion has no gradInput for class pred)
    self.gradInput[1]:resizeAs(input):zero()
    self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)

    -- learn the baseline reward
    self.gradInput[2] = self.criterion:backward(baseline, self.reward)
    self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)
    return self.gradInput
end

function GGClassReward0402:type(type)
    self._maxVal = nil
    self._maxIdx = nil
    self._target = nil
    local module = self.module
    self.module = nil
    local ret = parent.type(self, type)
    self.module = module
    return ret
end
