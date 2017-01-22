local CTmnist100, parent = torch.class("dp.CTmnist100", "dp.SmallImageSource")

function CTmnist100:__init(config)
   config = config or {}
   config.image_size = config.image_size or {1, 100, 100}
   config.name = config.name or 'CTmnist100'
   config.train_dir = config.train_dir or 'train'
   config.test_dir = 'test'
   config.download_url = config.download_url or '127.0.0.1/home/ryan/data/CTMnist100/CTMnist100.zip'
   parent.__init(self, config)
end
