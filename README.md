# SubMe
## Overview
- A speech-to-text API demonstrated with a subtitle generator application.
- Users can choose to get video/audio files transcribed with options on backend model or type of input file.
- Tech stack: Pytorch, Flask, React, Next.js, Google Cloud Service.

The server is available at https://subme-api.ue.r.appspot.com/

## Existing API endpoints

#### POST `/transcribe_file`
| Param | Type | Required | Description |
| ------------- | ------------- | -------------| -------------|
| dataset_name | string	| No	| Name of the dataset that should be used (LibriSpeech) |
| model_name	| string	| Yes	| Name of the model that should be used (quartznet, jasper) |
| file_datatype	| string	| Yes	| Type of file being sent to be processed - uploaded Audio file, uploaded Video file, or Youtube URL (a_upload, v_upload, y_video) |
| file | string/blob	| Yes	| The blob content of uploaded file or URL of Youtube video |


#### POST `/export_file`
| Param | Type | Required | Description |
| ------------- | ------------- | -------------| -------------|
| text | string	| Yes	| Transcription text to be exported to pdf format |
