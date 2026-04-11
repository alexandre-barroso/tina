import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhoneticLatentSpace(nn.Module):
    def __init__(self, dim=256, num_patterns=32):
        super(PhoneticLatentSpace, self).__init__()
        self.dim = dim
        self.num_patterns = num_patterns

        self.pattern_vectors = nn.Parameter(torch.randn(num_patterns, dim))

        self.register_buffer('usage_count', torch.zeros(num_patterns))
        self.register_buffer('phoneme_cooccurrence', torch.zeros(256, 256))
        self.register_buffer('context_sum', torch.zeros(num_patterns, dim))

        self.total_updates = 0

    def update_statistics(self, phoneme_seq, hidden_states):
        batch_size, seq_len = phoneme_seq.shape
        self.total_updates += batch_size

        for b in range(batch_size):
            for i in range(seq_len - 1):
                p1, p2 = phoneme_seq[b, i].item(), phoneme_seq[b, i + 1].item()
                if p1 < 256 and p2 < 256:
                    self.phoneme_cooccurrence[p1, p2] += 1

        with torch.no_grad():
            similarity = F.cosine_similarity(
                hidden_states.unsqueeze(1),
                self.pattern_vectors.unsqueeze(0),
                dim=2
            )

            top_patterns = similarity.argmax(dim=1)

            for pattern_idx in top_patterns:
                self.usage_count[pattern_idx] += 1
            for b in range(batch_size):
                pattern_idx = top_patterns[b].item()
                self.context_sum[pattern_idx] += hidden_states[b]

    def get_usage_statistics(self):
        return self.usage_count.clone().cpu().numpy()

    def get_average_context_vectors(self):
        avg_context = []
        for i in range(self.num_patterns):
            count = self.usage_count[i].item()
            if count > 0:
                avg = (self.context_sum[i] / count).detach().cpu().numpy()
                avg_context.append(avg)
            else:
                avg_context.append(None)
        return avg_context

    def reset_statistics(self):
        self.usage_count.zero_()
        self.phoneme_cooccurrence.zero_()
        self.context_sum.zero_()
        self.total_updates = 0

    def save_model(self, save_dir="./webapp/model", filename="latent_space_model.pth", epoch=None):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(self.state_dict(), save_path)
        if epoch is not None:
            print(f"Modelo de espaço latente salvo (época {epoch})")
        else:
            print(f"Modelo de espaço latente salvo")

if __name__ == '__main__':
    latent_space = PhoneticLatentSpace(dim=256, num_patterns=32)

    batch_size = 4
    seq_len = 10

    phoneme_seq = torch.randint(0, 256, (batch_size, seq_len))

    hidden_states = torch.randn(batch_size, latent_space.dim)

    latent_space.update_statistics(phoneme_seq, hidden_states)

    usage = latent_space.get_usage_statistics()
    avg_contexts = latent_space.get_average_context_vectors()

    print("Usage counts for each pattern vector:")
    print(usage)
    print("\nAverage context vectors (None indicates pattern not used):")
    for i, vec in enumerate(avg_contexts):
        print(f"Pattern {i}: {vec}")

    latent_space.save_model(epoch=1)
