require 'dp'
require 'dpnn'
require 'rnn'
require 'nn'
require 'optim'
require 'GGClassReward0402'
require 'GGClassReward0524'
require 'RecurrentGG0316'
require 'image'
require 'CTMnist'
require 'CTmnist'
require 'CTmnist100'
require 'ReinforceLinear'
require 'ReinforceIdentical'
require 'KLDivCriterion'



-- References :
-- A. http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- B. http://incompleteideas.net/sutton/williams-92.pdf

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Evaluate a Recurrent Model for Visual Attention')
cmd:text('Options:')
cmd:option('--xpPath', '/home/ryan/save/CTMnist100/GG4.02c ghs=512,lhs=128/GG4.02c ghs=512,lhs=128.dat', 'path to a previously saved model')
cmd:option('--cuda', false, 'model was saved with cuda')
cmd:option('--evalTest', false, 'model was saved with cuda')
cmd:option('--stochastic', false, 'evaluate the model stochatically. Generate glimpses stochastically')
cmd:option('--dataset', 'TCMnist', 'which dataset to use : Mnist | TCMnist | etc')
cmd:option('--overwrite', false, 'overwrite checkpoint')
cmd:option('--th', 0.4, 'overwrite checkpoint')
cmd:text()
local opt = cmd:parse(arg or {})


-- check that saved model exists
assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')

if opt.cuda then
   require 'cunn'
end

xp = torch.load(opt.xpPath)
model = xp:model().module
-- torch.save('GG4.02c ghs=512,lhs=128 finetuned rho=8.net',model)
tester = xp:tester() or xp:validator() -- dp.Evaluator
tester:sampler()._epoch_size = nil
conf = tester:feedback() -- dp.Confusion
cm = conf._cm -- optim.ConfusionMatrix

print("Last evaluation of "..(xp:tester() and 'test' or 'valid').." set :")
print(cm)

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

ra = model:findModules('nn.RecurrentGG0316')[1]
sg = model:findModules('nn.SpatialGlimpse')[1]

-- stochastic or deterministic
for i=1,#ra.actions do
   local rn = ra:findModules('nn.ReinforceNormal')[1]
   rn.stochastic = opt.stochastic
end

if opt.evalTest then
   conf:reset()
   tester:propagateEpoch(ds:testSet())

   print((opt.stochastic and "Stochastic" or "Deterministic") .. "evaluation of test set :")
   print(cm)
end

inputs = ds:get('test','inputs')
targets = ds:get('test','targets', 'b')

input = inputs:narrow(1,101,10)
target =  targets:narrow(1,101,10)
model:training() -- otherwise the rnn doesn't save intermediate time-step states
if not opt.stochastic then
   for i=1,#ra.actions do
      local rn = ra:findModules('nn.ReinforceNormal')[1]
      rn.stdev = 0 -- deterministic
   end
end
output = model:forward(input)
classval,classpred=output[1]:max(2)
classpred = classpred:double()


for i=1,target:size(1) do
  -- print(correct)
  if classpred[i]:max()~=target[i] then
    print(i)
    print(classpred[i]:max(),target[i])
  end
end

function drawBox(img, bbox, channel)
    channel = channel or 1

    local x1, y1 = torch.round(bbox[1]), torch.round(bbox[2])
    local x2, y2 = torch.round(bbox[1] + bbox[3]), torch.round(bbox[2] + bbox[4])

    x1, y1 = math.max(1, x1), math.max(1, y1)
    x2, y2 = math.min(img:size(3), x2), math.min(img:size(2), y2)

    local max = img:max()

    for i=x1,x2 do
        img[channel][y1][i] = max
        img[channel][y2][i] = max
    end
    for i=y1,y2 do
        img[channel][i][x1] = max
        img[channel][i][x2] = max
    end

    return img
end

locations = ra.actions
glanceNetwork =  torch.load('mnistConv100_Relu.net').module
for i=1,#glanceNetwork .modules do
    if tostring(glanceNetwork .modules[i]) == 'nn.LogSoftMax' then
        table.remove(glanceNetwork .modules, i)
    end
    if tostring(glanceNetwork .modules[i]) == 'nn.Convert' then
        table.remove(glanceNetwork .modules, i)
    end
    if tostring(glanceNetwork .modules[i]) == 'nn.Dropout' then
        glanceNetwork .modules[i].train = false
        glanceNetwork .modules[i].updateGradInput = function(input, gradOutput)
            return model.modules[i].gradInput
        end
    end
end


input = nn.Convert(ds:ioShapes(),'bchw'):forward(input)
glimpses = {}
patches = {}
glances = {}

params = nil
for i=1,input:size(1) do
   local img = input[i]
   local cs =  glanceNetwork:forward(input[i])
   -- print(cs)
   local dedo = torch.zeros(cs:size(1))
   dedo[target[i]] = -1
   local w = glanceNetwork:backward(input[i],dedo)-- dSc/dI
  --  w = (-w):cmax(0)
   w = w:abs()
   w = w:div(w:max())
   w = w:cmax(opt.th):add(-opt.th)
   for j,location in ipairs(locations) do
      local glimpse = glimpses[j] or {}
      glimpses[j] = glimpse
      local patch = patches[j] or {}
      patches[j] = patch
      local glance = glances[j] or {}
      glances[j] = glance

      local xy = location[i]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      -- (1,1), (input:size(3), input:size(4))
      x, y = x*(input:size(3)-1)+1, y*(input:size(4)-1)+1

      local gimg = img:clone()
      local glanceImg = w:clone()
      for d=1,sg.depth do
         local size = sg.size*(sg.scale^(d-1))
         local bbox = {y-size/2, x-size/2, size, size}
         drawBox(gimg, bbox, 1)
        --  glanceImg = drawBox(glanceImg:reshape(img:size()), bbox, 1)
      end
      glimpse[i] = gimg
      glance[i] = glanceImg


      local sg_, ps
      if j == 1 then
        sg_ = ra.rnn.initialModule:findModules('nn.SpatialGlimpse')[1]
      else
         sg_ = ra.rnn.sharedClones[j]:findModules('nn.SpatialGlimpse')[1]
      end
      patch[i] = image.scale(img:clone():float(), sg_.output[i]:narrow(1,1,1):float())

      collectgarbage()
   end
end

paths.mkdir('glimpse')
for j,glimpse in ipairs(glimpses) do
   local g = image.toDisplayTensor{input=glimpse,nrow=10,padding=3}
   local gl = image.toDisplayTensor{input=glances[j],nrow=10,padding=3}
   local p = image.toDisplayTensor{input=patches[j],nrow=10,padding=3}
   image.save("glimpse/glimpse"..j..".png", g)
   image.save("glimpse/glance"..j..".png", gl)
   image.save("glimpse/patch"..j..".png", p)
end
