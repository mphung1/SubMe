import React from "react";
import SearchBar from "@components/Youtube/YoutubeSearchBar";
import youtube from "@pages/api/youtube";
import VideoList from "@components/Youtube/VideoList";
import VideoDetail from "@components/Youtube/VideoDetail";
import { Center } from "@chakra-ui/react";

class Search extends React.Component<any> {
  state = {
    videos: [],
    selectedVideo: null,
  };

  handleSubmit = async (termFromSearchBar) => {
    const response = await youtube.get("/search", {
      params: {
        q: termFromSearchBar,
      },
    });

    this.setState({
      videos: response.data.items,
    });
  };

  handleVideoSelect = (video) => {
    this.setState({ selectedVideo: video });
    const videoId = video.id.videoId;
    this.props.searchCallback(videoId);
  };

  render() {
    return (
      <>
        <Center mt={2}>
          <SearchBar handleFormSubmit={this.handleSubmit} />
        </Center>
        <Center>
          <VideoDetail
            video={this.state.selectedVideo}
            renderContentScreen={this.props.renderContentScreen}
          />
        </Center>
        <Center mt={20}>
          <VideoList
            handleVideoSelect={this.handleVideoSelect}
            videos={this.state.videos}
          />
        </Center>
      </>
    );
  }
}
export default Search;
