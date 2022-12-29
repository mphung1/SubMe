import { Grid, Text } from "@chakra-ui/react";
import CustomButton from "@components/Fixed/CustomButton";

const VideoDetail = ({ video, renderContentScreen }) => {
  if (!video) {
    return (
        <Text color="gray">Your chosen video will be rendered here.</Text>
    );
  }

  const videoSrc = `https://www.youtube.com/embed/${video.id.videoId}`;

  return (
    <div>
    <Grid templateColumns='repeat(1, 1fr)' gap={2}>
      <iframe
        src={videoSrc}
        width="100%"
        height="300"
        allowFullScreen
        title="Video player"
      />
      <b>{video.snippet.title}</b>
      <p>
        <b>Description: </b>
        {video.snippet.description}
      </p>
      <CustomButton btnText="Proceed" onClick={renderContentScreen} />
    </Grid>
    </div>
  );
};

export default VideoDetail;
