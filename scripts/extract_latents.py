"""Extract actual Wan2.2 VAE and UMT5 latents, without loading/training transformer."""
import argparse,gc,json,pathlib
import numpy as np
import torch
import av
from lingbot_ur5.paths import ROOT,model_path
from lingbot_ur5.dataset import CAMERAS

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default=str(ROOT/'datasets/ur5_move_can_pot'))
    p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/pretrained/lingbot-va-base'))
    p.add_argument('--device',default='cuda');a=p.parse_args();root=pathlib.Path(a.dataset).resolve();model=pathlib.Path(a.checkpoint).resolve()
    model_path()
    from modules.utils import load_tokenizer,load_text_encoder,load_vae,WanVAEStreamingWrapper
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    info=json.loads((root/'meta/info.json').read_text());episodes=[json.loads(l) for l in (root/'meta/episodes.jsonl').read_text().splitlines()]
    texts=sorted({s['action_text'] for e in episodes for s in e['action_config']}|{''})
    tokenizer=load_tokenizer(str(model/'tokenizer'));encoder=load_text_encoder(str(model/'text_encoder'),torch.bfloat16,'cpu')
    embeddings={}
    for text in texts:
        inputs=tokenizer([prompt_clean(text)],padding='max_length',max_length=512,truncation=True,return_tensors='pt')
        emb=encoder(inputs.input_ids,inputs.attention_mask).last_hidden_state[0].to(torch.bfloat16)
        length=int(inputs.attention_mask.sum());emb[length:]=0;embeddings[text]=emb.cpu()
    torch.save(embeddings[''],root/'empty_emb.pt');del encoder,tokenizer;gc.collect()
    vae=load_vae(str(model/'vae'),torch.bfloat16,a.device);stream=WanVAEStreamingWrapper(vae)
    for episode in episodes:
        idx=episode['episode_index'];chunk=idx//info.get('chunks_size',1000)
        for key in CAMERAS.values():
            video=root/info['video_path'].format(episode_chunk=chunk,episode_index=idx,video_key=key)
            with av.open(str(video)) as container:frames=[f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
            for seg in episode['action_config']:
                start,end=seg['start_frame'],seg['end_frame'];ids=np.arange(start,end,4);ids=ids[:1+4*((len(ids)-1)//4)]
                if len(ids)<5:raise ValueError('Insufficient video frames for VAE')
                output=root/f'latents/chunk-{chunk:03d}'/key/f'episode_{idx:06d}_{start}_{end}.pth'
                if output.exists():continue
                stream.clear_cache();latents=[]
                for offset in range(0,len(ids)):
                    if offset!=0 and (offset-1)%4!=0:continue
                    group=ids[offset:offset+(1 if offset==0 else 4)]
                    video_tensor=torch.from_numpy(np.stack([frames[i] for i in group])).permute(3,0,1,2)[None].to(device=a.device,dtype=torch.bfloat16)/127.5-1
                    encoded=stream.encode_chunk(video_tensor);mu,_=encoded.chunk(2,dim=1)
                    mean=torch.tensor(vae.config.latents_mean,device=a.device).view(1,-1,1,1,1)
                    std=torch.tensor(vae.config.latents_std,device=a.device).view(1,-1,1,1,1)
                    latents.append(((mu.float()-mean)/std).to(torch.bfloat16).cpu())
                latent=torch.cat(latents,dim=2)[0];c,f,h,w=latent.shape
                assert f==(len(ids)-1)//4+1
                payload={'latent':latent.permute(1,2,3,0).reshape(-1,c).contiguous(),'latent_num_frames':f,'latent_height':h,'latent_width':w,
                        'video_num_frames':len(ids),'video_height':frames[0].shape[0],'video_width':frames[0].shape[1],
                        'text_emb':embeddings[seg['action_text']],'text':seg['action_text'],'frame_ids':ids.tolist(),
                        'start_frame':start,'end_frame':end,'fps':6.25,'ori_fps':25}
                output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,output);print('LATENT SAVED',output,tuple(latent.shape),flush=True)
    print('LATENT EXTRACTION COMPLETE')
if __name__=='__main__':main()
