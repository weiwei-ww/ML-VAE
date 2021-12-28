import logging
import models.CRDNN_CTC.model as CRDNN_CTC

logger = logging.getLogger(__name__)


class SBModel(CRDNN_CTC.SBModel):

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)

        wavs, wav_lens = batch['wav']
        feats = self.modules['wav2vec2'](wavs)  # (B, T, 1024)

        out = self.modules.crdnn(feats)
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        predictions = {
            'pout': pout,
        }

        return predictions
