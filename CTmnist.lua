--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

require 'torch'
require 'dp'
require 'torchx' -- for paths.indexdir
require 'image'


-- Copies values from src to dst.
local function update(dst, src)
  for k, v in pairs(src) do
    dst[k] = v
  end
end

-- Copies the config. An error is raised on unknown params.
local function updateDefaults(dst, src)
  for k, v in pairs(src) do
    if dst[k] == nil then
      error("unsupported param: " .. k)
    end
  end
  update(dst, src)
end

local function loadDataset(path)
  print(path)
  local dataset = torch.load(path)
  dataset.data = dataset.data:type(torch.Tensor():type())
  collectgarbage()
  dataset.data:mul(1/dataset.data:max())

  if dataset.data[1]:dim() ~= 3 then
    local sideLen = math.sqrt(dataset.data[1]:nElement())
    dataset.data = dataset.data:view(dataset.data:size(1), 1, sideLen, sideLen)
  end

  assert(dataset.labels:min() == 0, "expecting zero-based labels")
  return dataset
end

-- Return a list with pointers to selected examples.
local function selectSamples(examples, nSamples)
  local nExamples = examples:size(1)
  local samples = {}
  for i = 1, nSamples do
    samples[i] = examples[torch.random(1, nExamples)]
  end
  return samples
end

-- Puts the sprite on a random position inside of the obs.
-- The observation should have intensities in the [0, 1] range.
local function placeSpriteRandomly(obs, sprite, border)
  assert(obs:dim() == 3, "expecting an image")
  assert(sprite:dim() == 3, "expecting a sprite")
  local h = obs:size(2)
  local w = obs:size(3)
  local spriteH = sprite:size(2)
  local spriteW = sprite:size(3)

  local y = torch.random(1 + border, h - spriteH + 1 - border)
  local x = torch.random(1 + border, w - spriteW + 1 - border)

  local subTensor = obs[{{}, {y, y + spriteH - 1}, {x, x + spriteW - 1}}]
  subTensor:add(sprite)
  -- Keeping the values in the [0, 1] range.
  subTensor:apply(function(x)
    if x > 1 then
      return 1
    end
    if x < 0 then
      return 0
    end
    return x
  end)
end

local function placeDistractors(config, patch, examples)
  local distractors = selectSamples(examples, config.num_dist)
  local dist_w = config.dist_w
  local megapatch_w = config.megapatch_w

  local t_y, t_x, s_y, s_x
  for ind, d_patch in ipairs(distractors) do
    t_y = torch.random((megapatch_w-dist_w)+1)-1
    t_x = torch.random((megapatch_w-dist_w)+1)-1
    s_y = torch.random((d_patch:size(2)-dist_w)+1)-1
    s_x = torch.random((d_patch:size(3)-dist_w)+1)-1
    patch[{{}, {t_y+1,t_y+dist_w}, {t_x+1,t_x+dist_w}}]:add(d_patch[{{}, {s_y+1,s_y+dist_w}, {s_x+1,s_x+dist_w}}])
  end
  patch[patch:ge(1)]=1
end

-- Returns a map from {smallerDigit, biggerOrEqualDigit}
-- to an input in the softmax output.
local function createIndexMap(n, k)
  assert(k == 2, "expecting k=2")
  local indexMap = torch.Tensor(n, n):fill(0/0)
  local nextIndex = 1
  for i = 1, n do
    for j = i, n do
      indexMap[i][j] = nextIndex
      nextIndex = nextIndex + 1
    end
  end
  assert(k == 2 and nextIndex - 1 == (n * (n + 1))/2, "wrong count for k=2")
  return indexMap
end

local targetFilling = {}
function targetFilling.mark(target, usedClasses, config)
  -- The used encoding:
  --   target[digit + 1] will be 1 if the zero-based digit is present.
  target:resize(config.nClasses)
    :zero()
  for _, class in ipairs(usedClasses) do
    target[class] = 1
  end
end

function targetFilling.combine(target, usedClasses, config)
  -- We will have one softmax output for each
  -- combination-with-repetion of the two possible digits.
  local nClasses = config.nClasses
  local nOutputs = 1
  for k = 1, config.nDigits do
    nOutputs = nOutputs * (nClasses + k - 1) / k
  end
  target:resize(nOutputs)
    :zero()
  config.indexMap = config.indexMap or createIndexMap(nClasses, config.nDigits)
  assert(config.indexMap:max() == nOutputs, "wrong nOutputs")
  table.sort(usedClasses)
  target[config.indexMap[usedClasses]] = 1
end

function targetFilling.sum(target, usedClasses, config)
  local maxValue = (config.nClasses - 1) * config.nDigits
  -- The possible sums are {0, 1, ..., maxValue}
  target:resize(1 + maxValue)
    :zero()
  if config.nDigits == 2 then
    assert(target:nElement() == 19, "expecting 19 targets")
  end
  local value = torch.Tensor(usedClasses):add(-1):sum()
  assert(value >= 0 and value <= maxValue, "wrong sum")
  target[1 + value] = 1
end

-- The task is a classification of MNIST digits.
-- Each training example has a MNIST digit placed on a bigger black background.
function CTMnist(extraConfig)
  local config = {
    datasetPath = {'mnist/train.t7','mnist/test.t7'},
    -- The size of the background.
    megapatch_w = 28,
    -- Number of distractors.
    num_dist = 0,
    -- The distractor width.
    dist_w = 8,
    -- The width of a black border.
    border = 0,
    -- The number of digits in on image.
    nDigits = 1,
    -- The number of digit classes.
    nClasses = 10,

    -- The digits can be combined into one target for a softmax.
    -- Or the digits can be summed together.
    -- Otherwise the target should be modeled by Bernoulli units.
    targetFilling = "mark",
  }
  updateDefaults(config, extraConfig)

  local dataset_train = loadDataset(config.datasetPath[1])
  local dataset_test = loadDataset(config.datasetPath[2])

  local nTrainExamples = dataset_train.data:size(1)
  local perm = torch.Tensor()
  local nTestExamples = dataset_test.data:size(1)

  local obs = torch.Tensor(dataset_train.data[1]:size(1), config.megapatch_w, config.megapatch_w)
  assert(dataset_train.labels:max() < config.nClasses, "expecting labels from {0, .., nClasses - 1}")

  local target_train = torch.Tensor()
  local target_test = torch.Tensor()
  local fillTarget = assert(targetFilling[config.targetFilling], "unknown targetFilling")
  local step = nExamples

    local train_shuffle = torch.randperm(nTrainExamples) -- shuffle the data
    local train_input = torch.FloatTensor(nTrainExamples, 1, config.megapatch_w, config.megapatch_w)
    local train_target = torch.IntTensor(nTrainExamples)

    local test_shuffle = torch.randperm(nTestExamples) -- shuffle the data
    local test_input = torch.FloatTensor(nTestExamples, 1, config.megapatch_w, config.megapatch_w)
    local test_target = torch.IntTensor(nTestExamples)

    local obs = torch.Tensor(dataset_train.data[1]:size(1), config.megapatch_w, config.megapatch_w)

    print(config)

    for i=1,nTrainExamples do
      obs:zero()
      placeDistractors(config, obs, dataset_train.data)
      local idx = train_shuffle[i]
      sprite = dataset_train.data[i]
      placeSpriteRandomly(obs, sprite, config.border)
      train_input[idx]:copy(obs)
      train_target[idx] = dataset_train.labels[i]
      collectgarbage()
    end


    for i=1,nTestExamples do
      obs:zero()
      placeDistractors(config, obs, dataset_test.data)
      local idx = test_shuffle[i]
      sprite = dataset_test.data[i]
      placeSpriteRandomly(obs, sprite, config.border)
      test_input[idx]:copy(obs)
      test_target[idx] = dataset_test.labels[i]
      collectgarbage()
    end

    print("Saving train images to disk.")
    local trainDir = '/home/ryan/data/CTMnist100/train/'
    local labelsDirs = {}
    for i=0, 9 do
      labelsDir = paths.concat(trainDir, tostring(i))
      os.execute("mkdir " .. labelsDir)
      labelsDirs[i+1] = labelsDir
    end
    local indx = 1
    for i=1, nTrainExamples do
      image.save(paths.concat(tostring(labelsDirs[train_target[i]+1]),
                                  tostring(indx)..".png"), train_input[i])
      indx =indx +1
    end

    print("Saving test images to disk.")
    local testDir = '/home/ryan/data/CTMnist100/test/'
    labelsDirs = {}
    for i=0, 9 do
      labelsDir = paths.concat(testDir, tostring(i))
      os.execute("mkdir " .. labelsDir)
      labelsDirs[i+1] = labelsDir
    end
    for i=1, nTestExamples do
      image.save(paths.concat(tostring(labelsDirs[test_target[i]+1]),
                                  tostring(indx)..".png"), test_input[i])
      indx =indx +1
    end

    train_target:add(1)
    test_target:add(1)
    --
    -- local class={0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    --
    -- local trainInput = dp.ImageView('bchw', train_input)
    -- local trainTarget = dp.ClassView('b', train_target)
    -- local testInput = dp.ImageView('bchw', test_input)
    -- local testTarget = dp.ClassView('b', test_target)
    --
    -- trainTarget:setClasses({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    -- testTarget:setClasses({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    --
    -- local train = dp.DataSet{inputs=trainInput,targets=trainTarget,which_set='train'}
    -- local test = dp.DataSet{inputs=testInput,targets=testTarget,which_set='test'}
    --
    -- local ds = dp.DataSource{train_set=train,test_set=test}
    -- ds:classes{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    -- return ds
end
