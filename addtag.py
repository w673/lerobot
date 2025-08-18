from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("Ww1313w/TransferCube_Insertion", tag="v2.1", repo_type="dataset")
