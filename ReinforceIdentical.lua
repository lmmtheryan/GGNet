------------------------------------------------------------------------
--[[ ReinforceIdentical ]]--
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are mean (mu) of multivariate normal distribution.
-- Ouputs are samples drawn from these distributions.
-- Standard deviation is provided as constructor argument.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.237-239) which is
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceIdentical, parent = torch.class("nn.ReinforceIdentical", "nn.Reinforce")

function ReinforceIdentical:__init()
  parent.__init(self, stochastic)
  --  self.stdev = stdev
  --  if not stdev then
  --     self.gradInput = {torch.Tensor(), torch.Tensor()}
  --  end
  self.maxReward = 0
end

function ReinforceIdentical:updateOutput(input)
   self.output = input
   return self.output
end

function ReinforceIdentical:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : normal probability density function
   -- x : the sampled values (self.output)
   -- u : mean (mu) (mean)
   -- s : standard deviation (sigma) (stdev)

   local reward = self:NVRrewardAs(gradOutput)
  --  print(reward)
  --  assert(1==2)
  --  if reward:max()>self.maxReward then
  --    self.maxReward = reward:max()
  --  end
  --  print(self.maxReward)
  --  assert(1==2)
   self.gradInput:resizeAs(reward)
   self.gradInput:copy(reward)
  --  self.gradInput:copy(reward)
  --  self.gradInput:div(self.maxReward):cmax(0.5)
   self.gradInput:cmul(gradOutput)
  --  self.gradInput = gradOutput
   return self.gradInput
end
