


class Generator:
    def __init__(
        self,
        model: HyenaLM,
        encoder: Callable[[str], list[int]],
        decoder: Callable[[list[int]], str],
        max_seq_len: int,
        bos: int,
        eos: int,
        device: Any = torch.device("cpu")
    ):
        self.model = model.to(device=device)
        self.encoder = encoder
        self.decoder = decoder
        self.max_seq_len = max_seq_len
        self.bos = bos
        self.eos = eos
        self.device = device

    def generate(
        self,
        prompt: str,
        output_len: int,
        k: int = 10,
        temperature: float = 1.0,
        num_repeat: int = 2
    ) -> str:
        # L: seq len, V: vocab size, K: k
        self.model.eval()
        tokens = self.encoder(prompt)
        tokens.insert(0, self.bos)
        while len(tokens) < output_len + 1:
            x = torch.unsqueeze(
                torch.tensor(tokens[-self.max_seq_len:],
                             dtype=torch.long, device=self.device), 0
            )  # -> (1, L)
            logits = self.model(x)[0, -1, :]  # -> (V)

            values, indices = torch.topk(logits, k)  # -> (K) for each
            probas = torch.full_like(logits, float("-inf"))  # -> (V)
            probas.scatter_(0, indices, values)
            probas = fn.softmax(probas / temperature, dim=-1)  # (V) -> (V)

            next_token = torch.multinomial(probas, 1).item()
            # Check for repetition and if it's too much, redraw the token.
            count = 0
            for i in range(1, num_repeat + 1):
                if len(tokens) - i < 0 or tokens[-i] != next_token:
                    break
                count += 1
            while count >= num_repeat:
                next_token = torch.multinomial(probas, 1).item()
                count = 0
                for i in range(1, num_repeat + 1):
                    if len(tokens) - i < 0 or tokens[-i] != next_token:
                        break
                    count += 1

            tokens.append(next_token)
            print(self.decoder(next_token), end="")
        output = self.decoder(tokens)
        return output


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_):
        self.data = data
        self.tokenizer = tokenizer
        self.max_ = max_
        self.cash = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            self.tokenizer.sep_token_id = 5
            d = self.data[idx]

            input = self.tokenizer.encode_plus(d, truncation=True,
                                               max_length=self.max_, return_tensors="pt")  # Do not pad here

            input_ids = input["input_ids"]
            input_ids = (input_ids[0][:input_ids.size(1)-1]).unsqueeze(0)
            return {"input_ids": input_ids[0]}
            answer = f"{out}"

            input = self.tokenizer.encode_plus(question, truncation=True,
                                               max_length=self.max_, padding="max_length", return_tensors="pt")  # Do not pad here
            target_ids = self.tokenizer.encode_plus(answer, truncation=True,
                                                    max_length=self.max_, padding="max_length", return_tensors="pt")["input_ids"]  # Do not pad here

            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]

            return {"input_ids": input_ids[0], "attention_mask": attention_mask[0], "labels": target_ids[0], "text": question}
        except Exception as e:
            print(e)
            return {"input_ids": torch.zeros(self.max_length).int(), "attention_mask": torch.zeros(self.max_length).int(), "labels": torch.zeros(self.max_length).int()}


tokenizer = transformers.AutoTokenizer.from_pretrained(
    'rinna/japanese-gpt-1b', use_fast=False)
# データを整形


def split_text(text, length):
    words = text.split(' ')
    chunks = []
    chunk = []
    chunk_len = 0
    for word in words:
        if chunk_len + len(word) + 1 > length:  # The "+1" accounts for the space
            chunks.append(' '.join(chunk))
            chunk = [word]
            chunk_len = len(word)
        else:
            chunk.append(word)
            chunk_len += len(word) + 1
    chunks.append(' '.join(chunk))
    return chunks


tqdm.pandas()

embed_dim = 2048
vocab_size = tokenizer.vocab_size
depth = 24
config = HyenaConfig(
    embed_dim=embed_dim,
    max_seq_len=max_seq_len,
    activation="gelu"
)
model = HyenaLM(vocab_size, depth, hyena_config=config).cuda().bfloat16()


df_ = pd.read_parquet('text.parquet')


df_ = df_[~df_['text'].str.contains(
    '#質問|#vrchat|zabuu|#shindanmaker|#DLsite', case=False, regex=True) & ~df_['text'].str.startswith('RT ')]

# 質問箱 #VRChat
# df2 = [s for s in a if (len(s)>64 and len(s) <= 128) ]
# df = sorted(df, key=len)
df_ = list(df_["text"])
df_ = sorted(df_, key=len)
df_ = [s for s in df_ if (len(s) > 8)]
# dataset = QADataset(df, tokenizer,  max_=max_seq_len)
dataset_ = QADataset(df_, tokenizer,  max_=34)


def collate_fn(batch):

    input_ids = [item['input_ids'][:-1] for item in batch]
    input_ids2 = [item['input_ids'][1:] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=4)
    input_ids2 = pad_sequence(input_ids2, batch_first=True, padding_value=4)

    return {
        'input_ids': input_ids,
        'input_ids2': input_ids2,
        #    'attention_mask': torch.stack(padded_attention_mask).long(),
        #   'labels': torch.stack(padded_labels).long(),
    }


model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


loss_fn = nn.CrossEntropyLoss(ignore_index=4, label_smoothing=0.1)
dataloader_ = DataLoader(
    dataset_, batch_size=16*12,    collate_fn=collate_fn, shuffle=True)

# dataloader = DataLoader(
#   dataset, batch_size=40*3,    collate_fn=collate_fn,shuffle=True)
# dataloader2 = DataLoader(
#   dataset2, batch_size=16*4,    collate_fn=collate_fn)
epochs = 1


def encoder_fn(text):
    return tokenizer.encode(text, add_special_tokens=False)


def decoder_fn(tokens):
    return tokenizer.decode(tokens)


generator = Generator(
    model=model,
    encoder=encoder_fn,
    decoder=decoder_fn,
    max_seq_len=max_seq_len,
    bos=tokenizer.bos_token_id,
    eos=tokenizer.eos_token_id,
    device=torch.device("cuda") if torch.cuda.is_available(
    ) else torch.device("cpu")  # use cuda if available
)


loss_dir = "loss_values"
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
# Get the current timestamp to use in the filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
loss_filepath = os.path.join(loss_dir, f"loss_values_{timestamp}.csv")
model.train()
optimizer = Lion(
    model.parameters(), lr=6e-5, weight_decay=0.0001)

with open(loss_filepath, 'a', newline='') as loss_file:
    loss_writer = csv.writer(loss_file)

    for e in range(10):
        i = 0

        for epoch in range(epochs):
            optimizer.defaults["lr"]
            optimizer.zero_grad()
            tq_ = tqdm(dataloader_)  # 勾配累積なし
            optimizer.zero_grad()
            for data in tq_:
                try:
                    if i < 200:
                        lr = 6e-5*(0.005*i)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 6e-5
                    inputs = data["input_ids"].cuda()
                    inputs2 = data["input_ids2"].cuda()
                    logits = model(inputs)
                    loss = loss_fn(logits.transpose(1, 2), inputs2)
                    torch.cuda.empty_cache()
                    loss.backward()
                    tq_.set_postfix(
                        {"loss": loss.item(), "len": inputs[0].size()})
                    loss_writer.writerow([loss.item()])
                    loss_file.flush()
                    os.fsync(loss_file.fileno())
                    torch.cuda.empty_cache()
                    optimizer.step()  # スケーラーと一緒にオプティマイザのstepを実行
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                except:
                    print("err")
            #      except:
            #         print("err")
                if i % 128 == 0:
                    print()
                    str = tokenizer.decode(
                        inputs[0], skip_special_tokens=True)
                    tokens = tokenizer.encode(str)
                    _t = tokens[:len(tokens)//3]
                    _t = tokenizer.decode(_t)
                    print(str)
                    str = str[:len(str)//3]
                    print(
                        _t)
                    with torch.no_grad():
                        s = generator.generate(
                            prompt=_t,     output_len=64,     k=20,     temperature=0.4, )
                    print("\n")
                if i % 10000 == 0:
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, f"model_epoch_{e}_0_{epoch}_{i}.pth"))
                i = i+1

#
#        for epoch in range(epochs):
   #         i = 0
   #         tq = tqdm(dataloader)### 勾配累積なし
#
   #         optimizer.zero_grad()
   #         for data in tq:
   #             try:
   #                 inputs = data["input_ids"].cuda()
   #                 inputs2 = data["input_ids2"].cuda()
   #                 logits = model(inputs)
   #                 loss = loss_fn(logits.transpose(1, 2), inputs2)
   #                 torch.cuda.empty_cache()
   #                 loss.backward()
   #                 tq.set_postfix({"loss": loss.item(),"len":inputs[0].size()})
   #                 loss_writer.writerow([loss.item()])
   #                 loss_file.flush()
   #                 os.fsync(loss_file.fileno())
   #                 torch.cuda.empty_cache()
#
   #                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   #                 if i%8==0:
   #                     optimizer.step()
   #                     optimizer.zero_grad()
   #                     torch.cuda.empty_cache()
   #                     for param in model.parameters():
   #                         param.data = torch.clamp(param.data, -10, 10)
#
   #             except:
   #                 print("err")
#
#
   #      #       except:
   #       #          print("err")
   #             if i % 128 == 0:
   #                 print()
   #                 str=tokenizer.decode(inputs[0],skip_special_tokens=True)
#
   #                 str=str[:len(str)//2]
   #                 print(
   #                     str)
   #                 with torch.no_grad():
   #                     s = generator.generate(
   #                         prompt=str,     output_len=100,     k=20,     temperature=1.0, )
#
   #                 print("\n")
   #             if i % 10000 == 0:
   #                 torch.save(model.state_dict(), os.path.join(
   #                     model_dir, f"model_epoch_{e}_1_{epoch}_{i}.pth"))
   #             i = i+1
#